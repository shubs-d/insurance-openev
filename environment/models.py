from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class ClaimAction(str, Enum):
    auto_approve = "auto_approve"
    assign_tier1_adjuster = "assign_tier1_adjuster"
    assign_specialist_adjuster = "assign_specialist_adjuster"
    request_more_documents = "request_more_documents"
    escalate_fraud_review = "escalate_fraud_review"


class ClaimObservation(BaseModel):
    claim_id: str
    claim_type: str
    claim_amount: float
    policy_age_days: int
    claimant_age: int
    incident_description: str
    documents_submitted: List[str]
    prior_claims_count: int
    fraud_risk_score: float
    location_risk_score: float
    claim_urgency: str


class EnvironmentState(BaseModel):
    current_claim: ClaimObservation
    steps_taken: int = 0
    done: bool = False
    total_reward: float = 0.0
    last_action: Optional[str] = None
    # Track effective state changes from actions without mutating the claim
    effective_docs: Optional[int] = None
    effective_fraud_score: Optional[float] = None


class RewardOutput(BaseModel):
    reward: float
    done: bool
    info: Optional[dict] = None
