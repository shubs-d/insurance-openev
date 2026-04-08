from environment.env import InsuranceClaimsEnv
from environment.scenarios import generate_scenarios, classify_scenario
from environment.models import ClaimAction


# -------------------------
# Baseline Agent
# -------------------------

def baseline_agent(obs):

    fraud = obs[0]
    amount = obs[1] * 20000
    docs = obs[2] * 5

    if fraud > 0.8:
        return ClaimAction.escalate_fraud_review

    elif fraud > 0.6:
        return ClaimAction.assign_specialist_adjuster

    elif docs < 1:
        return ClaimAction.request_more_documents

    elif fraud < 0.25 and amount < 5000:
        return ClaimAction.auto_approve

    else:
        return ClaimAction.assign_tier1_adjuster


# -------------------------
# Safe Average
# -------------------------

def safe_avg(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0.0


# -------------------------
# Main Evaluation
# -------------------------

def main():

    print("\nRunning RL Evaluation...\n")

    scenarios = generate_scenarios()

    scores = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    for i, claim in enumerate(scenarios):

        print(f"\nCase {i+1}")

        env = InsuranceClaimsEnv(claim)

        obs, _ = env.reset()

        total_reward = 0
        done = False

        while not done:

            action = baseline_agent(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            print(
                f" Step {env.state.steps_taken} -> {action.value} | reward {reward:.2f}"
            )

            total_reward += reward

        difficulty = classify_scenario(claim)

        score = max(0.0, min(1.0, total_reward))

        scores[difficulty].append(score)

        print(  
            f"{difficulty.upper()} | "
            f"Fraud: {claim.fraud_risk_score:.2f} | "
            f"Amt: {claim.claim_amount} | "
            f"Docs: {len(claim.documents_submitted)} | "
            f"Final Score: {score:.2f}"
        )

    easy_avg = safe_avg(scores["easy"])
    medium_avg = safe_avg(scores["medium"])
    hard_avg = safe_avg(scores["hard"])

    final_score = (easy_avg + medium_avg + hard_avg) / 3

    print("\n--------------------------")

    print(f"Easy Avg Score: {easy_avg:.2f}")
    print(f"Medium Avg Score: {medium_avg:.2f}")
    print(f"Hard Avg Score: {hard_avg:.2f}")

    print("\n--------------------------")

    print(f"Final Score: {final_score:.2f}")

    print("--------------------------\n")


if __name__ == "__main__":
    main()