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
environment/
    env.py
    models.py
    scenarios.py

tasks/
    easy_task.py
    medium_task.py
    hard_task.py

graders/
    grader.py

inference.py
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
- Low fraud probability
- Low claim amount
- Sufficient documentation

### Medium Scenarios
- Moderate fraud probability
- Mixed signals
- Some ambiguity

### Hard Scenarios
- Conflicting signals
- High claim amount
- Limited documentation
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

Each claim contains:

- Fraud risk score
- Claim amount
- Number of documents

Observation vector:

```
[fraud_score, claim_amount, num_documents]
```

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

- Correct approval
- Fraud detection
- Efficient routing

Negative rewards:

- Wrong approval
- Unnecessary escalation
- Poor routing

---

## Example

```
Case 1

Step 1 -> auto_approve | reward 1.00

Final Score: 1.00
```

---

## Gym Compatibility

Environment follows Gymnasium API:

- reset()
- step()
- observation_space
- action_space

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

---

## Requirements

```
pydantic
gymnasium
numpy
```

---

## Evaluation

We test:

- 60 scenarios
- Easy
- Medium
- Hard

Final Score = Average Performance

---

## Hackathon Requirements

- RL Environment ✓
- Reward shaping ✓
- Multi-step decisions ✓
- Agent baseline ✓
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
