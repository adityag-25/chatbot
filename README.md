# 🚀 Autonomous Customer Support Decision Agent (OpenEnv)

## 🧠 Overview

This project implements a **real-world simulation environment** for training and evaluating autonomous customer support agents.

Unlike traditional chatbots, this system enables:

* Multi-step decision making
* Context-aware reasoning
* Tool usage simulation
* Emotion-aware responses

---

## ⚙️ Key Features

### ✅ Multi-turn Conversations

The agent interacts over multiple steps with evolving state.

### ✅ Structured Action Space

Each action includes:

* Intent classification
* Decision making
* Response generation

### ✅ Emotion Awareness

System detects user emotion (angry, confused, neutral) and adjusts evaluation.

### ✅ Tool Usage Simulation

Supports backend actions:

* Refund processing
* Ticket escalation
* Order lookup

### ✅ Advanced Reward Function

Reward is computed using:

* Intent correctness
* Decision accuracy
* Semantic similarity (LLM-based)
* Emotional alignment
* Efficiency (fewer turns)

---

## 📊 Tasks

* 30+ real-world scenarios
* Easy → Medium → Hard
* Includes ambiguous and emotional queries

---

## 🧪 Evaluation

Score range: **0.0 – 1.0**

Components:

* Intent: 25%
* Decision: 35%
* Response Quality: 25%
* Emotion Handling: 10%
* Efficiency: 5%

---

## 🚀 API Endpoints

* `POST /reset`
* `POST /step`
* `GET /state`

---

## 🤖 Baseline Agent

Includes an LLM-powered baseline agent using OpenAI API.

---

## 🐳 Deployment

Dockerized and ready for HuggingFace Spaces.

---

## 🏆 Why This Project Stands Out

* Real-world applicability (customer support automation)
* Multi-step reasoning (not just text generation)
* Strong evaluation methodology
* Modular and extensible design

---

## 🔧 How to Run

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

---

## 📈 Future Work

* Reinforcement learning integration
* Real-time API integrations
* Multi-agent collaboration

---

## 👨‍💻 Author

Hackathon Submission - Scaler OpenEnv Challenge
