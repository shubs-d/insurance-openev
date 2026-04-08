import random
import uuid
from .models import ClaimObservation


CLAIM_TYPES = ["auto", "health", "property"]

URGENCY = ["low", "medium", "high"]

DOCUMENT_POOL = [
    "id_proof",
    "invoice",
    "medical_report",
    "police_report",
    "photos",
]


def generate_claim(difficulty):

    claim_type = random.choice(CLAIM_TYPES)

    if difficulty == "easy":

        fraud = random.uniform(0.0, 0.3)
        amount = random.uniform(500, 5000)
        docs = random.sample(DOCUMENT_POOL, random.randint(2, 4))
        prior = random.randint(0, 1)

    elif difficulty == "medium":

        fraud = random.uniform(0.3, 0.7)
        amount = random.uniform(3000, 10000)
        docs = random.sample(DOCUMENT_POOL, random.randint(1, 3))
        prior = random.randint(0, 2)

    else:  # hard

        fraud = random.uniform(0.5, 0.95)
        amount = random.uniform(8000, 20000)
        docs = random.sample(DOCUMENT_POOL, random.randint(0, 2))
        prior = random.randint(1, 4)

    return ClaimObservation(
        claim_id=str(uuid.uuid4()),
        claim_type=claim_type,
        claim_amount=amount,
        policy_age_days=random.randint(30, 1000),
        claimant_age=random.randint(18, 75),
        incident_description="Generated claim scenario",
        documents_submitted=docs,
        prior_claims_count=prior,
        fraud_risk_score=fraud,
        location_risk_score=random.uniform(0.0, 1.0),
        claim_urgency=random.choice(URGENCY)
    )


def generate_scenarios():

    scenarios = []

    for _ in range(20):
        scenarios.append(generate_claim("easy"))

    for _ in range(20):
        scenarios.append(generate_claim("medium"))

    for _ in range(20):
        scenarios.append(generate_claim("hard"))

    random.shuffle(scenarios)

    return scenarios

def classify_scenario(claim):

    fraud = claim.fraud_risk_score
    amount = claim.claim_amount
    docs = len(claim.documents_submitted)

    if fraud < 0.3 and amount < 5000 and docs >= 2:
        return "easy"

    elif fraud < 0.7:
        return "medium"

    else:
        return "hard"