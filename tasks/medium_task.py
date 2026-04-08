from environment.models import ClaimObservation


def medium_claim() -> ClaimObservation:
    return ClaimObservation(
        claim_id="medium_001",
        claim_type="property",
        claim_amount=5000.0,
        policy_age_days=200,
        claimant_age=42,
        incident_description="Water leakage damage",
        documents_submitted=["photo"],
        prior_claims_count=1,
        fraud_risk_score=0.4,
        location_risk_score=0.5,
        claim_urgency="medium"
    )