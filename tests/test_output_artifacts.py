from pathlib import Path
from PIL import Image
import pytest

TASK2_DIR = Path("images/task2")
TASK3_DIR = Path("images/task3")


def _assert_valid_image(img_path: Path, expected_size=(1024, 1024)):
    """Helper to assert image file is non-empty, readable, and standard RGB/RGBA."""
    assert img_path.stat().st_size > 10_000, f"File {img_path.name} is suspiciously small (<10KB)"
    with Image.open(img_path) as img:
        assert img.format in ("PNG", "JPEG")
        assert img.mode in ("RGB", "RGBA")
        if expected_size:
            assert img.size == expected_size, f"{img_path.name} size {img.size} != {expected_size}"


def test_task2_pipeline_artifacts():
    """Validates Task 2 concept images and final composite presentation board."""
    if not TASK2_DIR.exists():
        pytest.skip("images/task2/ not found; skipping offline artifact check.")

    concepts = list((TASK2_DIR / "concepts").glob("*.png"))
    assert len(concepts) > 0, "No concept images found in images/task2/concepts/"
    for img_path in concepts[:3]:
        _assert_valid_image(img_path, expected_size=(1024, 1024))

    board_path = TASK2_DIR / "final_concepts_presentation.png"
    if board_path.exists():
        assert board_path.stat().st_size > 20_000
        with Image.open(board_path) as img:
            assert img.width >= 1024 and img.height >= 512


def test_task3_agent_seasonal_artifacts():
    """Validates Task 3 agentic concept assets and stitched seasonal board."""
    if not TASK3_DIR.exists():
        pytest.skip("images/task3/ not found; skipping agent artifact check.")

    concepts = list((TASK3_DIR / "concepts").glob("*.png"))
    if concepts:
        for img_path in concepts[:3]:
            _assert_valid_image(img_path, expected_size=(1024, 1024))

    boards = list(TASK3_DIR.glob("presentation_*.png"))
    if boards:
        with Image.open(boards[0]) as img:
            assert img.width >= 1024