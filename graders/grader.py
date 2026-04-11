from environment.models import ClaimAction, ClaimObservation


def safe_score(x: float) -> float:
    EPS = 1e-2
    if x <= 0.0:
        print(f"[BAD SCORE DETECTED] {x}", flush=True)
        return EPS
    if x >= 1.0:
        print(f"[BAD SCORE DETECTED] {x}", flush=True)
        return 1.0 - EPS
    return x


def grade_claim(observation: ClaimObservation, action: ClaimAction) -> float:

    fraud = observation.fraud_risk_score
    amount = observation.claim_amount
    docs = observation.documents_submitted

    raw = 0.2  # default

    # Fraud scenario
    if fraud > 0.8:
        if action == ClaimAction.escalate_fraud_review:
            raw = 0.99
        elif action == ClaimAction.assign_specialist_adjuster:
            raw = 0.6
        else:
            raw = 0.01

    # Missing docs
    elif len(docs) < 2:
        if action == ClaimAction.request_more_documents:
            raw = 0.99
        elif action == ClaimAction.assign_tier1_adjuster:
            raw = 0.5
        else:
            raw = 0.01

    # High amount
    elif amount > 10000:
        if action == ClaimAction.assign_specialist_adjuster:
            raw = 0.99
        elif action == ClaimAction.assign_tier1_adjuster:
            raw = 0.6
        else:
            raw = 0.01

    # Clean claim
    elif action == ClaimAction.auto_approve:
        raw = 0.99

    return safe_score(raw)
