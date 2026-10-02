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