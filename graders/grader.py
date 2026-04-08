from environment.models import ClaimAction, ClaimObservation


def grade_claim(observation: ClaimObservation, action: ClaimAction) -> float:

    fraud = observation.fraud_risk_score
    amount = observation.claim_amount
    docs = observation.documents_submitted

    # Fraud scenario
    if fraud > 0.8:
        if action == ClaimAction.escalate_fraud_review:
            return 1.0
        elif action == ClaimAction.assign_specialist_adjuster:
            return 0.6
        else:
            return 0.0

    # Missing docs
    if len(docs) < 2:
        if action == ClaimAction.request_more_documents:
            return 1.0
        elif action == ClaimAction.assign_tier1_adjuster:
            return 0.5
        else:
            return 0.0

    # High amount
    if amount > 10000:
        if action == ClaimAction.assign_specialist_adjuster:
            return 1.0
        elif action == ClaimAction.assign_tier1_adjuster:
            return 0.6
        else:
            return 0.0

    # Clean claim
    if action == ClaimAction.auto_approve:
        return 1.0

    return 0.2