"""
Upgraded baseline runner — supports both heuristic and OpenAI-powered agents.

Usage
-----
  # Heuristic agent (no API key needed)
  python baseline/run.py

  # OpenAI agent
  OPENAI_API_KEY=sk-... python baseline/run.py

The script now reads obs.conversation for decision-making (not an empty list)
and surfaces dynamic order_id / order_status from each task.
"""

import os
import sys
import json

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import CustomerSupportEnv, Action


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

def heuristic_agent(obs) -> Action:
    """Simple rule-based agent that reads from obs.conversation."""
    text = " ".join(m.content for m in obs.conversation).lower()

    if "refund" in text:
        return Action(
            intent="refund_request",
            decision="refund",
            response=(
                "I sincerely apologize for the trouble. "
                "Your refund has been processed and will appear in 3-5 business days."
            ),
        )
    elif any(w in text for w in ["broken", "wrong", "rude", "empty", "stuck",
                                   "late", "not received", "hasn't arrived"]):
        return Action(
            intent="complaint",
            decision="escalate",
            response=(
                "We are truly sorry for this experience. "
                "I have escalated your case to our priority support team "
                "who will contact you within 24 hours."
            ),
        )
    elif any(w in text for w in ["delay", "angry", "frustrated", "worst"]):
        return Action(
            intent="complaint",
            decision="escalate",
            response=(
                "We sincerely apologize for the inconvenience. "
                "Your case has been escalated and our team will resolve this promptly."
            ),
        )
    elif any(w in text for w in ["address", "order id", "details", "change", "cancel"]):
        return Action(
            intent="modify_order",
            decision="ask_info",
            response=(
                "Please share your order ID and any relevant details "
                "so I can assist you right away."
            ),
        )
    elif any(w in text for w in ["track", "where", "package", "delivery", "status"]):
        return Action(
            intent="track_order",
            decision="resolve",
            response=(
                "Your order is in transit. "
                "You can track it using the link in your confirmation email. "
                "Estimated delivery: 3-5 business days."
            ),
        )
    else:
        return Action(
            intent="info_request",
            decision="resolve",
            response=(
                "Thank you for contacting us! "
                "I'm happy to help. Your request has been noted and "
                "will be resolved shortly."
            ),
        )


# ---------------------------------------------------------------------------
# OpenAI agent
# ---------------------------------------------------------------------------

def openai_agent(obs, task: dict, api_key: str) -> Action:
    """OpenAI-powered agent with structured JSON output."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=api_key)

    conversation_text = "\n".join(
        f"{m.role}: {m.content}" for m in obs.conversation
    )
    system_prompt = (
        "You are an autonomous customer support agent.\n"
        "Valid intents: refund_request, complaint, info_request, track_order, "
        "modify_order, cancel_request, payment_issue, technical_issue, "
        "return_request, exchange_request, refund_issue, request\n"
        "Valid decisions: resolve, refund, escalate, ask_info\n\n"
        'Respond ONLY with JSON: {"intent": "...", "decision": "...", "response": "..."}'
    )
    user_prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        f"Customer emotion: {obs.emotion}\n"
        f"Order status: {obs.context.get('order_status', 'unknown')}\n"
        "Provide your action:"
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


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(num_tasks: int = 5):
    api_key = os.getenv("OPENAI_API_KEY", "")
    agent_name = "OpenAI GPT" if api_key else "Heuristic"
    print(f"\n{'='*60}")
    print(f"  OpenEnv Baseline Simulation — {agent_name} Agent")
    print(f"{'='*60}\n")

    env = CustomerSupportEnv()
    total_reward = 0.0

    for task_num in range(num_tasks):
        print(f"--- Task {task_num + 1} ---")
        obs = env.reset()

        # Print opening state
        print(f"Query      : {obs.conversation[0].content}")
        print(f"Emotion    : {obs.emotion}")
        print(f"Order ID   : {obs.context.get('order_id', 'N/A')}")
        print(f"Order Status: {obs.context.get('order_status', 'N/A')}")

        done = False
        turn = 0
        episode_reward = 0.0

        while not done:
            # Choose agent
            if api_key:
                try:
                    action = openai_agent(obs, env.current_task, api_key)
                except Exception as e:
                    print(f"  [OpenAI error: {e}] — falling back to heuristic")
                    action = heuristic_agent(obs)
            else:
                action = heuristic_agent(obs)

            print(f"\n  Turn {turn + 1}:")
            print(f"    Intent   : {action.intent}")
            print(f"    Decision : {action.decision}")
            print(f"    Response : {action.response[:80]}...")

            result = env.step(action)
            obs = result.observation
            done = result.done
            episode_reward = result.reward
            turn += 1

            print(f"    Reward   : {result.reward:.4f}")
            if result.info.get("tool_result"):
                tr = result.info["tool_result"]
                print(f"    Tool     : [{tr.get('tool', '')}] {tr.get('message', '')}")

        total_reward += episode_reward
        print(f"\n  Episode reward: {episode_reward:.4f}\n")

    avg = total_reward / num_tasks
    print(f"{'='*60}")
    print(f"  Average Score: {avg:.4f}   ({num_tasks} tasks)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_simulation(num_tasks=5)
