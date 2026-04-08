import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .models import ClaimObservation, ClaimAction, EnvironmentState


class InsuranceClaimsEnv(gym.Env):

    def __init__(self, claim: ClaimObservation):
        super().__init__()

        # Store the original so reset() can always return to a clean slate
        self._original_claim = claim.model_copy(deep=True)
        self.claim = claim.model_copy(deep=True)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(ClaimAction))

        self.state = EnvironmentState(
            current_claim=self.claim,
            steps_taken=0,
            done=False
        )

    def _get_obs(self):
        return np.array([
            self.claim.fraud_risk_score,
            self.claim.claim_amount / 20000,
            len(self.claim.documents_submitted) / 5,
            self.claim.location_risk_score,
            self.claim.prior_claims_count / 10
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Deep-copy the original claim so every episode starts completely clean
        self.claim = self._original_claim.model_copy(deep=True)

        self.state = EnvironmentState(
            current_claim=self.claim,
            steps_taken=0,
            done=False
        )

        return self._get_obs(), {}

    def step(self, action):

        reward = 0.0

        self.state.steps_taken += 1

        # Read current values — FIX: do not mutate self.claim inside step()
        fraud = self.claim.fraud_risk_score
        amount = self.claim.claim_amount
        docs = len(self.claim.documents_submitted)

        # --------------------------------
        # Penalize repeated actions
        # --------------------------------
        if self.state.last_action == action:
            reward -= 0.15

        self.state.last_action = action

        # --------------------------------
        # Action effects
        # --------------------------------

        if action == ClaimAction.request_more_documents:
            # Reward requesting docs only when genuinely sparse
            reward += 0.3 if docs < 2 else -0.1
            # Track the effective doc count in state, not by mutating claim
            self.state.effective_docs = docs + 1

        elif action == ClaimAction.assign_specialist_adjuster:
            # FIX: store effective fraud score in state instead of on claim object
            self.state.effective_fraud_score = min(1.0, fraud + 0.05)
            reward += 0.6 if fraud > 0.5 or amount > 10000 else -0.2

        elif action == ClaimAction.assign_tier1_adjuster:
            reward += 0.5 if fraud < 0.6 else -0.2

        elif action == ClaimAction.escalate_fraud_review:
            reward += 0.85 if fraud > 0.7 else -0.3
            self.state.done = True

        elif action == ClaimAction.auto_approve:
            reward += 1.0 if fraud < 0.25 and amount < 5000 else -0.5
            self.state.done = True

        # --------------------------------
        # Step penalty (encourage faster decisions)
        # --------------------------------
        reward -= 0.05

        # --------------------------------
        # Max 3 steps
        # --------------------------------
        if self.state.steps_taken >= 3:
            self.state.done = True

        self.state.total_reward += reward

        return (
            self._get_obs(),
            reward,
            self.state.done,
            False,
            {}
        )
