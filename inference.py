import argparse
import os

from agents.baseline import llm_agent, rule_based_agent
from environment.env import InsuranceClaimsEnv
from environment.scenarios import generate_scenarios, classify_scenario


# -------------------------
# Safe Average
# -------------------------

def safe_avg(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0.0


def validate_llm_environment(agent_mode: str):
    required = ["OPENAI_API_KEY", "MODEL_NAME"]
    missing_required = [name for name in required if not os.getenv(name)]

    if missing_required:
        print(
            "[WARN] Missing environment variables for LLM mode: "
            + ", ".join(missing_required),
            flush=True,
        )

    # HF_TOKEN is allowed as a key fallback for OpenAI-compatible HF endpoints.
    has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN"))
    has_model_name = bool(os.getenv("MODEL_NAME"))

    if agent_mode == "llm" and (not has_api_key or not has_model_name):
        missing = []
        if not has_api_key:
            missing.append("OPENAI_API_KEY or HF_TOKEN")
        if not has_model_name:
            missing.append("MODEL_NAME")

        print(
            "[ERROR] LLM agent selected but missing configuration: "
            + ", ".join(missing),
            flush=True,
        )
        return False

    return True


def select_action(obs, agent_mode: str):
    if agent_mode == "llm":
        return llm_agent(obs)

    return rule_based_agent(obs)


# -------------------------
# Main Evaluation
# -------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["rule_based", "llm"],
        default="rule_based",
        help="Select inference policy",
    )
    args = parser.parse_args()

    if not validate_llm_environment(args.agent):
        return 1

    print("\nRunning RL Evaluation...\n")

    scenarios = generate_scenarios()

    scores = {
        "easy": [],
        "medium": [],
        "hard": []
    }

    for i, claim in enumerate(scenarios):

        difficulty = classify_scenario(claim)

        print(f"\nCase {i+1}")
        print(f"[START] task={difficulty}", flush=True)

        env = InsuranceClaimsEnv(claim)

        obs, _ = env.reset()

        total_reward = 0
        done = False
        step_num = 0

        while not done:

            action = select_action(obs, args.agent)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            step_num += 1

            print(
                f" Step {env.state.steps_taken} -> {action.value} | reward {reward:.2f}"
            )
            print(f"[STEP] step={step_num} reward={reward}", flush=True)

            total_reward += reward

        score = max(0.0, min(1.0, total_reward))

        scores[difficulty].append(score)

        print(  
            f"{difficulty.upper()} | "
            f"Fraud: {claim.fraud_risk_score:.2f} | "
            f"Amt: {claim.claim_amount} | "
            f"Docs: {len(claim.documents_submitted)} | "
            f"Final Score: {score:.2f}"
        )
        print(f"[END] task={difficulty} score={score} steps={step_num}", flush=True)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())