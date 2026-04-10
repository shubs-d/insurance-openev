from typing import Sequence

from environment.models import ClaimAction

from .llm_client import generate_action


def rule_based_agent(obs: Sequence[float]) -> ClaimAction:
    fraud = obs[0]
    amount = obs[1] * 20000
    docs = obs[2] * 5

    if fraud > 0.8:
        return ClaimAction.escalate_fraud_review

    if fraud > 0.6:
        return ClaimAction.assign_specialist_adjuster

    if docs < 1:
        return ClaimAction.request_more_documents

    if fraud < 0.25 and amount < 5000:
        return ClaimAction.auto_approve

    return ClaimAction.assign_tier1_adjuster


def build_llm_prompt(obs: Sequence[float]) -> str:
    fraud = float(obs[0])
    amount = float(obs[1]) * 20000
    docs = float(obs[2]) * 5
    location_risk = float(obs[3]) if len(obs) > 3 else 0.0
    prior_claims = float(obs[4]) * 10 if len(obs) > 4 else 0.0

    return (
        "Decide the best insurance claim action for this observation.\n"
        f"fraud_risk_score={fraud:.4f}\n"
        f"claim_amount={amount:.2f}\n"
        f"documents_submitted_count={docs:.2f}\n"
        f"location_risk_score={location_risk:.4f}\n"
        f"prior_claims_count={prior_claims:.2f}\n"
        "Return exactly one action token."
    )


def llm_agent(obs: Sequence[float]) -> ClaimAction:
    prompt = build_llm_prompt(obs)
    action_name = generate_action(prompt)
    return ClaimAction(action_name)
