"""
CustomerSupportEnv — the core OpenEnv simulation environment.

Fixes applied vs. original:
  - reset() seeds context["history"] with the initial user message
  - Dynamic order_id & order_status read from each task
  - order_lookup tool wired to "ask_info" and order-related "resolve" actions
  - Multi-turn follow-ups include the actual order_id from the task
"""

import json
import random

from .models import Observation, Action, StepResult
from .grader import grade_action
from .emotion import detect_emotion
from .tools import process_refund, escalate_ticket, order_lookup

MAX_TURNS = 3

# Intents that are related to order status and warrant an order_lookup call
ORDER_RELATED_INTENTS = {
    "track_order", "complaint", "refund_issue", "cancel_request", "modify_order",
}


class CustomerSupportEnv:
    def __init__(self, task_file: str = "tasks/tasks.json"):
        with open(task_file) as f:
            self.tasks = json.load(f)

        self.state = None
        self.current_task = None
        self.turn = 0
        self.done = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task_index: int = None) -> Observation:
        """
        Start a new episode.

        Args:
            task_index: If provided, picks that specific task from self.tasks
                        (useful for the benchmark runner). Otherwise random.
        """
        if task_index is not None:
            self.current_task = self.tasks[task_index % len(self.tasks)]
        else:
            self.current_task = random.choice(self.tasks)

        self.turn = 0
        self.done = False

        emotion = detect_emotion(self.current_task["query"])

        initial_message = {"role": "user", "content": self.current_task["query"]}

        self.state = {
            "conversation": [initial_message],
            "context": {
                # Dynamic values from the task; fallback to sensible defaults
                "order_id": self.current_task.get("order_id", f"ORD-{random.randint(1000, 9999)}"),
                "order_status": self.current_task.get("order_status", "unknown"),
                # Seed history with the opening user message (fixes multi-turn bug)
                "history": [initial_message],
                "task_index": self.tasks.index(self.current_task),
            },
            "emotion": emotion,
        }

        return Observation(**self.state)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode finished — call reset() to start a new one.")

        self.turn += 1

        # Append agent response to conversation
        agent_msg = {"role": "agent", "content": action.response}
        self.state["conversation"].append(agent_msg)

        # Keep the sliding-window history (last 5 messages) in context
        self.state["context"]["history"] = self.state["conversation"][-5:]

        # ---- Tool execution ------------------------------------------------
        tool_result: dict = {}
        order_id: str = self.state["context"]["order_id"]
        order_status: str = self.state["context"]["order_status"]

        if action.decision == "refund":
            tool_result = process_refund(order_id)

        elif action.decision == "escalate":
            tool_result = escalate_ticket(
                issue=self.current_task.get("query", "customer issue"),
                order_id=order_id,
            )

        elif action.decision in ("ask_info", "resolve"):
            # Look up order whenever the query is order-related
            if action.intent in ORDER_RELATED_INTENTS or "order" in action.intent.lower():
                tool_result = order_lookup(order_id, order_status)

        # ---- Multi-turn follow-up simulation --------------------------------
        if self.turn < MAX_TURNS and action.decision == "ask_info":
            follow_ups = [
                f"Here is my order ID: {order_id}",
                "I already shared details earlier, please check again.",
                "Why do you need that? Can you just fix it?",
            ]
            follow_up_msg = {"role": "user", "content": random.choice(follow_ups)}
            self.state["conversation"].append(follow_up_msg)
            self.state["context"]["history"] = self.state["conversation"][-5:]

        # ---- Grading -------------------------------------------------------
        reward = grade_action(action, self.current_task, self.turn)

        # ---- Termination ---------------------------------------------------
        if action.decision in ("refund", "resolve", "escalate"):
            self.done = True
        elif self.turn >= MAX_TURNS:
            self.done = True

        return StepResult(
            observation=Observation(**self.state),
            reward=reward,
            done=self.done,
            info={
                "tool_result": tool_result,
                "turn": self.turn,
                "task": self.current_task,
            },
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def state_fn(self) -> dict:
        """Return raw state dict (used by /state endpoint)."""
        return self.state
