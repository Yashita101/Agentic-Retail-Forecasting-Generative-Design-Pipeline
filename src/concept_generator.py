"""
concept_generator.py
Subtask 2 Module: Fashion Product Concept Generation Engine.
Defines interface for generative design inspired by high-performing styles.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

class BaseConceptGenerator(ABC):
    """
    Abstract interface for fashion concept generation to ensure loose coupling
    between orchestration (main) and specific generative backends.
    """
    def __init__(self, output_dir: str | Path = "images/task2/concepts") -> None:
        """Configures the output directory for generated imagery."""
        self._output_dir = Path(output_dir)

    @abstractmethod
    def generate_concept(self, prompt: str, neg_prompt: str, filename: str) -> Path:
        """
        Executes generation, persists result, and returns output path.
        """
        pass


class HFDiffusionConceptGenerator(BaseConceptGenerator):
    """
    Hugging Face inference engine utilizing Stable Diffusion for fashion text-to-image.
    Injects professional catalog aesthetics into generated outputs.
    """

    _DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        """
        Initializes InferenceClient. Requires 'HF_TOKEN' set in environment.
        Task 2 logic remains encapsulated here.
        """
        super().__init__()
        # Import delayed until init to keep Task 1 strictly analytics-only
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub package required for Task 2. Run: pip install huggingface_hub pillow"
            )

        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token or token == "your_token_here":
            # Will be raised even if Task 2 is disabled in main if main initiates generator.
            # Best handled by orchestration layer not initializing this if Task 2 is off.
            raise ValueError(
                "Valid 'HF_TOKEN' not found in .env. Task 2 execution blocked."
            )
        self._model_id = model_id
        self._client = InferenceClient(model=self._model_id, token=token)

    def generate_concept(self, prompt: str, neg_prompt: str, filename: str) -> Path:
        """
        Call HF Inference API, convert response to image, and save.
        Enforces SDXL 1024x1024 sizing and strict negative guardrails.
        """
        print(f"Subtask 2: Calling HF API via model {self._model_id}...")
        
        try:
            image: Image.Image = self._client.text_to_image(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=1024,
                height=1024
            )
            
            output_dir = Path(self._output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{filename}.png"
            image.save(output_path)
            print(f"Subtask 2: Concept image saved: {output_path.as_posix()}")
            return output_path
            
        except Exception as e:
            print(f"Error during concept generation: {e}")
            raise

def build_sdxl_prompt(product_type: str, color: str, season: str = "summer") -> tuple[str, str]:
    """Pure helper function formulating positive and negative prompts for SDXL fashion generation.
    
    Returns:
        tuple[str, str]: (positive_prompt, negative_prompt)
    """
    season_modifiers = {
        "winter": "Heavyweight insulating fabrics, structured winter-ready construction.",
        "spring": "Lightweight breathable fabrics, clean transitional tailoring.",
        "summer": "Airy open-weave fabric, fluid relaxed hot-weather drape.",
        "autumn": "Mid-weight textured twill, rich earthy layered styling."
    }
    modifier = season_modifiers.get(season.strip().lower(), "High-quality construction.")
    
    ptype_lower = product_type.lower()
    if any(w in ptype_lower for w in ["top", "shirt", "blouse", "sweater", "cardigan", "hoodie", "jacket", "waistcoat", "vest", "t-shirt"]):
        cut = "featuring a realistic hollow 3D neck cavity showing the inside back of the collar, isolated upper-body garment"
    elif any(w in ptype_lower for w in ["trouser", "short", "legging", "skirt"]):
        cut = "featuring a realistic hollow 3D open waistband cavity, isolated lower-body garment"
    elif any(w in ptype_lower for w in ["dress", "jumpsuit"]):
        cut = "featuring a realistic hollow 3D neck cavity, full-length isolated garment"
    else:
        cut = "featuring a hollow 3D opening, isolated single garment"

    prompt = (
        f"Commercial e-commerce product catalog shot of exactly ONE single isolated {color} {product_type}. "
        f"Strictly displayed on an invisible ghost mannequin, {cut}, centered front-facing view. "
        f"Basic commercial design modification with subtle refined stitching, preserving the original {product_type} silhouette exactly. "
        f"{modifier} High fashion commercial studio lighting, neutral light-grey backdrop, 8k resolution, photorealistic clothing item. "
        f"THIS IS A 3D GARMENT, NOT A FLAT PATTERN."
    )

    neg_prompt = (
        "ABSOLUTELY NO HUMAN MODEL, no face, no skin, no hands, no arms, no legs, no physical body inside the garment, "
        "no mannequin head, no hanger. fabric swatch, seamless pattern, wallpaper, flat 2d design, background texture, "
        "macro shot, close up, sketch, drawing, illustration, cartoon, 3d render, multiple views, multiple items, "
        "cropped, blurry, low quality, collage, storyboard."
    )

    return prompt, neg_prompt