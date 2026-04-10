import os
from typing import Iterable

from openai import OpenAI

from environment.models import ClaimAction


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _get_api_key() -> str | None:
    """Prefer OPENAI_API_KEY but allow HF_TOKEN for Hugging Face OpenAI-compatible endpoints."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")


def build_client() -> OpenAI:
    return OpenAI(
        api_key=_get_api_key(),
        base_url=os.getenv("API_BASE_URL", DEFAULT_BASE_URL),
    )


def _valid_actions_text(actions: Iterable[ClaimAction]) -> str:
    return ", ".join(a.value for a in actions)


def _extract_action(raw_text: str) -> str:
    response = raw_text.strip().lower()

    # Prefer exact match first, then robust substring fallback.
    for action in ClaimAction:
        if response == action.value:
            return action.value

    for action in ClaimAction:
        if action.value in response:
            return action.value

    valid = _valid_actions_text(ClaimAction)
    raise ValueError(f"Invalid LLM action '{raw_text}'. Expected one of: {valid}")


def generate_action(prompt: str) -> str:
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise ValueError("MODEL_NAME is required for LLM agent mode")

    client = build_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an insurance claims triage policy. "
                    "Return exactly one action token and no other text. "
                    f"Allowed actions: {_valid_actions_text(ClaimAction)}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    message = completion.choices[0].message.content or ""
    return _extract_action(message)
