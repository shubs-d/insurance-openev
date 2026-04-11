import os
from typing import Iterable

from openai import OpenAI

from environment.models import ClaimAction

def build_client() -> OpenAI:
    assert "API_KEY" in os.environ, "Missing API_KEY"
    assert "API_BASE_URL" in os.environ, "Missing API_BASE_URL"

    print("Using API_BASE_URL:", os.environ["API_BASE_URL"], flush=True)

    return OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
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
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = build_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an insurance claims decision agent.",
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n"
                    "Return exactly one action token and no other text. "
                    f"Allowed actions: {_valid_actions_text(ClaimAction)}"
                ),
            },
        ],
        temperature=0,
    )

    message = completion.choices[0].message.content or ""
    return _extract_action(message)
