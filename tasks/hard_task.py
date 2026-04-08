from environment.models import ClaimObservation


def hard_claim() -> ClaimObservation:
    return ClaimObservation(
        claim_id="hard_001",
        claim_type="health",
        claim_amount=18000.0,
        policy_age_days=45,
        claimant_age=29,
        incident_description="Emergency hospitalization claim",
        documents_submitted=["invoice"],
        prior_claims_count=4,
        fraud_risk_score=0.92,
        location_risk_score=0.7,
        claim_urgency="high"
    )