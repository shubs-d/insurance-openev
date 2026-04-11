---
sdk: docker
app_port: 8000
title: Insurance Claims Triage RL Environment
emoji: 🏦
colorFrom: blue
colorTo: green
pinned: false
---

# Insurance Claims RL Agent

## Overview

This project builds a Reinforcement Learning environment for automated insurance claim processing.

The goal is to simulate real-world insurance decision-making using RL agents.

The system evaluates claims and chooses appropriate actions:

- Auto approve claim
- Assign Tier-1 adjuster
- Assign Specialist adjuster
- Request more documents
- Escalate fraud review

The agent receives rewards based on decision quality.

---

## Why This Matters

Insurance companies process thousands of claims daily.

Challenges:

- Fraud detection
- Cost control
- Efficient processing
- Risk management

Our RL agent helps:

- Reduce fraud losses
- Speed up approvals
- Optimize claim routing
- Reduce manual work

---

## Architecture

```
agents/
    baseline.py
    llm_client.py

environment/
    env.py
    models.py
    scenarios.py

server/
    app.py

tasks/
    easy_task.py
    medium_task.py
    hard_task.py

graders/
    grader.py

inference.py
openenv.yaml
requirements.txt
Dockerfile
README.md
```

## Scenario Generation

We use semi-realistic scenario generation based on real insurance patterns.

The generator models real-world claim characteristics:

- High claim amount increases fraud probability
- Fewer submitted documents increases uncertainty
- Multiple prior claims increase risk
- High-risk locations increase fraud probability
- Claim urgency influences routing decisions

We generate three difficulty levels:

### Easy Scenarios
- Low fraud probability (0.0 – 0.3)
- Low claim amount ($500 – $5,000)
- Sufficient documentation (2–4 docs)

### Medium Scenarios
- Moderate fraud probability (0.3 – 0.7)
- Mixed signals
- Some ambiguity

### Hard Scenarios
- High fraud probability (0.5 – 0.95)
- High claim amount ($8,000 – $20,000)
- Limited documentation (0–2 docs)
- High prior claims

This approach allows:

- Realistic simulation
- Scalable evaluation
- Robust RL training

---

## RL Environment

We built a custom Gym-compatible environment:

```
InsuranceClaimsEnv
```

The environment supports:

- reset()
- step()
- observation_space
- action_space

This allows training RL agents.

---

## Observations

Each claim is encoded as a 5-dimensional observation vector:

```
[
  fraud_risk_score,
  claim_amount / 20000,
  documents_submitted / 5,
  location_risk_score,
  prior_claims_count / 10
]
```

All values are normalized to the [0, 1] range.

---

## Actions

Agent can choose:

- auto_approve
- assign_tier1_adjuster
- assign_specialist_adjuster
- request_more_documents
- escalate_fraud_review

---

## Reward Logic

Positive rewards:

- Correct approval of clean claims (+1.0)
- Fraud escalation on high-risk claims (+0.85)
- Specialist assignment for complex claims (+0.6)
- Document request when docs are sparse (+0.3)

Negative rewards:

- Wrong approval of risky claims (-0.5)
- Unnecessary escalation (-0.3)
- Poor routing (-0.2)
- Step penalty per action (-0.05)
- Repeated action penalty (-0.15)

All final task scores are clamped to the open interval (0.01, 0.99).

---

## Example

```
Case 1

Step 1 -> auto_approve | reward 0.95

Final Score: 0.9500
```

---

## Gym Compatibility

Environment follows Gymnasium API:

- reset()
- step()
- observation_space (Box, shape=(5,))
- action_space (Discrete, 5 actions)

This enables RL training.

---

## Running

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python inference.py
```

Run with explicit agent mode:

```
python inference.py --agent rule_based
python inference.py --agent llm
```

---

## Environment Variables

The LLM agent uses the evaluator's OpenAI-compatible proxy.

- `API_KEY`: required proxy API key
- `API_BASE_URL`: required proxy base URL
- `MODEL_NAME`: optional model identifier (defaults to `gpt-4o-mini`)

Evaluator proxy example:

```bash
export API_KEY=your_proxy_key
export API_BASE_URL=https://your-proxy-base-url/v1
export MODEL_NAME=gpt-4o-mini
python inference.py --agent llm
```

`inference.py` defaults to `--agent llm` so at least one real proxy call is made during evaluation.

---

## Requirements

```
pydantic
gymnasium
numpy
openai
```

---

## Evaluation

We test:

- 60 scenarios (20 easy, 20 medium, 20 hard)
- Final Score = Average of (Easy Avg + Medium Avg + Hard Avg) / 3

---

## Hackathon Requirements

- RL Environment ✓
- Reward shaping ✓
- Multi-step decisions ✓
- Agent baseline (rule-based + LLM) ✓
- Evaluation system ✓
- Gym compatibility ✓
- Docker support ✓

---

## Future Work

- Train RL agents
- Multi-agent workflows
- Real dataset integration
- Online learning

---

## Team

Scaler Hackathon Submission

Insurance Claims RL Agent
