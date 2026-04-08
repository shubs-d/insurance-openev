from environment.models import ClaimObservation


def easy_claim() -> ClaimObservation:
    return ClaimObservation(
        claim_id="easy_001",
        claim_type="auto",
        claim_amount=1200.0,
        policy_age_days=400,
        claimant_age=35,
        incident_description="Minor bumper damage",
        documents_submitted=["invoice", "photo"],
        prior_claims_count=0,
        fraud_risk_score=0.1,
        location_risk_score=0.2,
        claim_urgency="low"
    )