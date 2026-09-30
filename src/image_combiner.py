"""
src/image_combiner.py
Combines seasonal concept images into a unified presentation board.
"""
from pathlib import Path
from PIL import Image


class ConceptImageCombiner:
    """Stitches individual garment renders into a horizontal presentation board."""

    def __init__(self, canvas_color: str = "#2b2b2b", padding: int = 40):
        self.canvas_color = canvas_color
        self.padding = padding

    def combine_seasonal_concepts(
        self,
        input_dir: str | Path,
        output_file: str | Path,
        season: str,
        expected_count: int = 3
    ) -> Path | None:
        """
        Combines concept images matching a season prefix.
        Gracefully handles missing files without crashing.
        """
        in_path = Path(input_dir)
        out_path = Path(output_file)

        if not in_path.exists():
            print(f"[ImageCombiner] ⚠️ Warning: Directory '{in_path}' does not exist. Skipping image stitching.")
            return None

        # Filter strictly by season prefix and valid image extensions
        valid_exts = {".png", ".jpg", ".jpeg"}
        season_prefix = f"agent_concept_{season.strip().lower()}_"
        
        matched_files = sorted([
            f for f in in_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts and f.name.startswith(season_prefix)
        ])

        if not matched_files:
            print(f"[ImageCombiner] ⚠️ No concept images found matching prefix '{season_prefix}' in '{in_path}'. Stitching skipped.")
            return None

        if len(matched_files) < expected_count:
            print(f"[ImageCombiner] ⚠️ Found {len(matched_files)} image(s) for season '{season}', expected {expected_count}. Proceeding with available images.")

        images_to_combine = matched_files[:expected_count]
        loaded_images = [Image.open(p) for p in images_to_combine]

        widths, heights = zip(*(img.size for img in loaded_images))
        total_width = sum(widths) + (self.padding * (len(loaded_images) + 1))
        max_height = max(heights) + (self.padding * 2)

        canvas = Image.new("RGB", (total_width, max_height), color=self.canvas_color)

        x_offset = self.padding
        for img in loaded_images:
            canvas.paste(img, (x_offset, self.padding))
            x_offset += img.width + self.padding

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        print(f"[ImageCombiner] ✅ Presentation board saved: {out_path.as_posix()}")
        return out_path

# Standalone CLI test
if __name__ == "__main__":
    combiner = ConceptImageCombiner()
    # Example for Task 2:
    # combiner.combine("images/task2/concepts", "images/task2/final_concept_presentation.png")