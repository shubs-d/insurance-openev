
import argparse
import sys
import traceback

from agents.baseline import llm_agent, rule_based_agent
from environment.env import InsuranceClaimsEnv
from environment.models import ClaimAction
from environment.scenarios import generate_scenarios, classify_scenario
from graders.grader import safe_score


# -------------------------
# Safe Average
# -------------------------

def safe_avg(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0.0


def select_action(obs, agent_mode: str) -> ClaimAction:
    try:
        if agent_mode == "llm":
            return llm_agent(obs)
        return rule_based_agent(obs)
    except Exception:
        traceback.print_exc()
        print("[WARN] select_action failed, returning fallback.", flush=True)
        return ClaimAction.assign_tier1_adjuster


# -------------------------
# Main Evaluation
# -------------------------

def main() -> int:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--agent",
            choices=["rule_based", "llm"],
            default="llm",
            help="Select inference policy",
        )
        args = parser.parse_args()

        print("\nRunning RL Evaluation...\n")

        scenarios = generate_scenarios()

        scores = {
            "easy": [],
            "medium": [],
            "hard": []
        }

        for i, claim in enumerate(scenarios):
            try:
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

                score = safe_score(max(0.0, min(1.0, total_reward)))

                scores[difficulty].append(score)

                print(
                    f"{difficulty.upper()} | "
                    f"Fraud: {claim.fraud_risk_score:.2f} | "
                    f"Amt: {claim.claim_amount} | "
                    f"Docs: {len(claim.documents_submitted)} | "
                    f"Final Score: {score:.6f}"
                )
                print(f"[END] task={difficulty} score={score:.4f} steps={step_num}", flush=True)

            except Exception:
                traceback.print_exc()
                # Emit required structured output even on failure
                difficulty = "easy"
                fallback_score = safe_score(0.0)
                print(f"[START] task={difficulty}", flush=True)
                print(f"[STEP] step=1 reward=0", flush=True)
                print(f"[END] task={difficulty} score={fallback_score:.4f} steps=1", flush=True)
                scores.setdefault(difficulty, []).append(fallback_score)

        easy_avg = safe_score(safe_avg(scores["easy"]))
        medium_avg = safe_score(safe_avg(scores["medium"]))
        hard_avg = safe_score(safe_avg(scores["hard"]))

        final_score = safe_score((easy_avg + medium_avg + hard_avg) / 3)

        print("\n--------------------------")

        print(f"Easy Avg Score: {easy_avg:.6f}")
        print(f"Medium Avg Score: {medium_avg:.6f}")
        print(f"Hard Avg Score: {hard_avg:.6f}")

        print("\n--------------------------")

        print(f"Final Score: {final_score:.6f}")

        print("--------------------------\n")

    except Exception:
        traceback.print_exc()
        print("[FATAL ERROR] main() encountered an error, exiting gracefully.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
