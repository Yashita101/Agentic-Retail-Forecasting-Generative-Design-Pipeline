import pytest
from src.concept_generator import build_sdxl_prompt

GOLDEN_PROMPT_CASES = [
    {
        "input": {"product_type": "Trousers", "color": "Black", "season": "summer"},
        "must_include": ["trousers", "black", "airy open-weave fabric", "hollow 3d open waistband cavity", "invisible ghost mannequin"],
        "must_not_include": ["human model", "no face", "no skin", "no arms"]
    },
    {
        "input": {"product_type": "Hoodie", "color": "Dark Blue", "season": "winter"},
        "must_include": ["hoodie", "dark blue", "heavyweight insulating fabrics", "hollow 3d neck cavity"],
        "must_not_include": ["human model", "no physical body", "no mannequin head"]
    }
]

@pytest.mark.parametrize("case", GOLDEN_PROMPT_CASES)
def test_prompt_template_regression_suite(case):
    """Prompt Regression: Verifies engineered prompt adheres strictly to structural design constraints."""
    prompt, neg_prompt = build_sdxl_prompt(
        product_type=case["input"]["product_type"],
        color=case["input"]["color"],
        season=case["input"]["season"]
    )
    
    # Assert positive prompt invariants
    for keyword in case["must_include"]:
        assert keyword.lower() in prompt.lower(), f"Prompt failed to include required term: '{keyword}'"
        
    # Assert negative prompt security invariants
    for restricted in case["must_not_include"]:
        assert restricted.lower() in neg_prompt.lower(), f"Negative prompt missed guardrail: '{restricted}'"