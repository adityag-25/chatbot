"""
FastAPI server for the Autonomous Customer Support Decision Agent (OpenEnv).

Endpoints
---------
POST /reset          — Start a new episode (optional task_index body param)
POST /step           — Take one action in the current episode
GET  /state          — Return raw environment state
GET  /tasks          — Return all tasks from tasks.json
POST /agent/step     — Auto-step using the heuristic or OpenAI agent
POST /benchmark      — Run the agent across all tasks and return scores

CORS is fully open so the bundled HTML dashboard can reach the API
regardless of origin.
"""

import os
import json
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Allow running from project root (python -m uvicorn api.main:app …)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CustomerSupportEnv, Observation, Action, StepResult

app = FastAPI(
    title="OpenEnv — Customer Support Decision Agent",
    version="2.0.0",
    description="Interactive simulation environment for autonomous customer support agents.",
)

# ---------------------------------------------------------------------------
# CORS — allow the HTML dashboard (any origin) to call these endpoints
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance (single-session demo — good enough for hackathon)
# ---------------------------------------------------------------------------
env = CustomerSupportEnv(task_file="tasks/tasks.json")
_last_step_result: Optional[dict] = None  # cache latest StepResult for /state


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_index: Optional[int] = None


class StepRequest(BaseModel):
    intent: str
    decision: str
    response: str


class BenchmarkRequest(BaseModel):
    max_tasks: Optional[int] = None   # None → all tasks


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Serve the dashboard HTML if requested from a browser, else API info."""
    ui_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ui", "index.html"
    )
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"message": "OpenEnv API is running. See /docs for Swagger UI."}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    """Start a new episode, optionally pinning a specific task by index."""
    global _last_step_result
    _last_step_result = None
    obs = env.reset(task_index=body.task_index)
    return obs.model_dump()


@app.post("/step")
def step(body: StepRequest):
    """Take one agent action and receive the next observation + reward."""
    global _last_step_result
    if env.state is None
        raise HTTPException(status_code=400, detail="Call /reset first.")

    action = Action(intent=body.intent, decision=body.decision, response=body.response)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _last_step_result = result.model_dump()
    return _last_step_result


@app.get("/state")
def state():
    """Return the current raw environment state plus the latest step result."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return {
        "state: env.state_fn(),
        "last_step_result": _last_step_result,
        "turn": env.turn,
        "done": env.done,
        "task": env.current_task,
    }


@app.get("/tasks")
def get_tasks():
    """Return all tasks from tasks.json with their indices."""
    tasks_with_index = [
        {"index": i, **task} for i, task in enumerate(env.tasks)
    ]
    return {"tasks": tasks_with_index, "count": len(env.tasks)}


@app.post("/agent/step")
def agent_auto_step():
    """
    Auto-step using the OpenAI agent (if OPENAI_API_KEY is set) or the
    heuristic fallback agent.  Returns the StepResult identical to /step.
    """
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env.done:
        raise HTTPException(status_code=400, detail="Episode done — call /reset.")

    action = _get_agent_action(env)
    return step(StepRequest(
        intent=action.intent,
        decision=action.decision,
        response=action.response,
    ))


@app.post("/benchmark")
def run_benchmark(body: BenchmarkRequest = BenchmarkRequest()):
    """
    Run the heuristic agent across all (or max_tasks) tasks and return
    per-task scores plus aggregate statistics.
    """
    tasks = env.tasks
    limit = body.max_tasks if body.max_tasks else len(tasks)
    results = []

    for idx in range(min(limit, len(tasks))):
        bench_env = CustomerSupportEnv(task_file="tasks/tasks.json")
        obs = bench_env.reset(task_index=idx)
        done = False
        episode_reward = 0.0
        turns = 0
        tool_calls = []

        while not done:
            action = _get_agent_action(bench_env)
            result = bench_env.step(action)
            obs = result.observation
            done = result.done
            episode_reward = result.reward  # use last-turn reward
            turns += 1
            if result.info.get("tool_result"):
                tool_calls.append(result.info["tool_result"].get("tool", ""))

        task = bench_env.current_task
        results.append({
            "index": idx,
            "query": task["query"],
            "intent": task["intent"],
            "decision": task["decision"],
            "emotion": task.get("emotion", "neutral"),
            "reward": episode_reward,
            "turns": turns,
            "tool_calls": tool_calls,
            "pass": episode_reward >= 0.5,
        })

    rewards = [r["reward"] for r in results]
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    pass_rate = sum(1 for r in results if r["pass"]) / len(results) if results else 0.0

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "average_reward": round(avg, 4),
            "pass_rate": round(pass_rate, 4),
            "max_reward": round(max(rewards), 4) if rewards else 0,
            "min_reward": round(min(rewards), 4) if rewards else 0,
        },
    }


# ---------------------------------------------------------------------------
# Agent logic (heuristic + optional OpenAI)
# ---------------------------------------------------------------------------

def _get_agent_action(env_instance: CustomerSupportEnv) -> Action:
    """
    Try OpenAI if OPENAI_API_KEY is available; fall back to heuristic.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            return _openai_agent(env_instance, api_key)
        except Exception:
            pass  # graceful fallback

    return _heuristic_agent(env_instance)


def _heuristic_agent(env_instance: CustomerSupportEnv) -> Action:
    """
    Rule-based baseline agent that reads from the conversation history.
    """
    history = env_instance.state["context"]["history"]
    text = " ".join(m["content"] if isinstance(m, dict) else m.content
                    for m in history).lower()

    # Decision rules
    if "refund" in text:
        decision = "refund"
        intent = "refund_request"
        response = ("I sincerely apologize for the inconvenience. "
                    "Your refund has been initiated and will be credited "
                    "to your account within 3-5 business days.")
    elif any(w in text for w in ["broken", "wrong", "rude", "empty", "stuck",
                                  "late", "hasn't arrived", "not received"]):
        decision = "escalate"
        intent = "complaint"
        response = ("I am truly sorry to hear about your experience. "
                    "I have escalated your issue to our senior support team "
                    "who will contact you within 24 hours.")
    elif any(w in text for w in ["delay", "angry", "frustrated", "worst"]):
        decision = "escalate"
        intent = "complaint"
        response = ("We sincerely apologize for the inconvenience. "
                    "Your case has been escalated and our team will resolve this promptly.")
    elif any(w in text for w in ["order id", "address", "details", "change", "cancel"]):
        decision = "ask_info"
        intent = "modify_order"
        response = ("Please provide your order ID and any additional details "
                    "so I can assist you better.")
    elif any(w in text for w in ["track", "where", "package", "delivery", "status"]):
        decision = "resolve"
        intent = "track_order"
        response = ("Your order is currently being tracked. "
                    "It is in transit and should arrive within the estimated timeframe. "
                    "Please check the tracking link in your confirmation email.")
    else:
        decision = "resolve"
        intent = "info_request"
        response = ("Thank you for reaching out! "
                    "I am happy to help you today. "
                    "Your request has been noted and will be resolved shortly.")

    return Action(intent=intent, decision=decision, response=response)


def _openai_agent(env_instance: CustomerSupportEnv, api_key: str) -> Action:
    """
    OpenAI-powered agent.  Returns structured JSON parsed into an Action.
    Falls back to heuristic on any error.
    """
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)

    history = env_instance.state["context"]["history"]
    task = env_instance.current_task
    conversation_text = "\n".join(
        f"{m['role'] if isinstance(m, dict) else m.role}: "
        f"{m['content'] if isinstance(m, dict) else m.content}"
        for m in history
    )

    system_prompt = (
        "You are an autonomous customer support agent. "
        "Given the conversation, classify the customer's intent and decide the best action.\n\n"
        "Valid intents: refund_request, complaint, info_request, track_order, "
        "modify_order, cancel_request, payment_issue, technical_issue, "
        "return_request, exchange_request, refund_issue, request\n"
        "Valid decisions: resolve, refund, escalate, ask_info\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"intent": "<intent>", "decision": "<decision>", "response": "<your response to the customer>"}'
    )

    user_prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        f"Task context: emotion={task.get('emotion', 'neutral')}, "
        f"order_status={env_instance.state['context'].get('order_status', 'unknown')}\n\n"
        "Provide your action as JSON:"
    )

    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=200,
    )

    raw = chat.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)
    return Action(
        intent=parsed["intent"],
        decision=parsed["decision"],
        response=parsed["response"],
    )
