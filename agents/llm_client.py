import os
import traceback
from typing import Iterable

from openai import OpenAI

from environment.models import ClaimAction

# Default fallback action when anything goes wrong
_FALLBACK_ACTION = ClaimAction.assign_tier1_adjuster.value


def build_client() -> OpenAI:
    api_key = os.environ.get("API_KEY", "")
    api_base = os.environ.get("API_BASE_URL", "")

    if not api_key or not api_base:
        raise RuntimeError("Missing API_KEY or API_BASE_URL")

    print("Using API_BASE_URL:", api_base, flush=True)

    return OpenAI(
        api_key=api_key,
        base_url=api_base,
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

    # Instead of raising, return fallback
    print(
        f"[WARN] Could not parse LLM action '{raw_text}', using fallback.",
        flush=True,
    )
    return _FALLBACK_ACTION


def generate_action(prompt: str) -> str:
    """Make an API call via LiteLLM proxy; always returns a valid action string."""
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    try:
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

    except Exception:
        traceback.print_exc()
        print(
            f"[WARN] LLM call failed, returning fallback action: {_FALLBACK_ACTION}",
            flush=True,
        )
        return _FALLBACK_ACTION
