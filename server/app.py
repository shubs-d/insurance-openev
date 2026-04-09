import uvicorn
from fastapi import FastAPI, Body
from environment.env import InsuranceClaimsEnv
from environment.scenarios import generate_scenarios
from environment.models import ClaimAction

app = FastAPI(title="Insurance Claims Triage RL Environment")

env: InsuranceClaimsEnv | None = None


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Insurance Claims Triage Environment API is actively running",
        "docs": "Access /docs for the Swagger UI"
    }



@app.post("/reset")
def reset():
    global env
    scenarios = generate_scenarios()
    env = InsuranceClaimsEnv(scenarios[0])
    obs, info = env.reset()
    return {
        "observation": obs.tolist(),
        "info": info
    }


@app.post("/step")
def step(action: str = Body(..., embed=True)):
    action_enum = ClaimAction(action)
    obs, reward, terminated, truncated, info = env.step(action_enum)
    return {
        "observation": obs.tolist(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }


@app.get("/state")
def state():
    return {
        "steps": env.state.steps_taken,
        "done": env.state.done,
        "total_reward": env.state.total_reward,
        "last_action": env.state.last_action,
        "effective_docs": env.state.effective_docs,
        "effective_fraud_score": env.state.effective_fraud_score,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
