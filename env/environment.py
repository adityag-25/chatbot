import json
import random

from .models import Observation, Action, StepResult
from .grader import grade_action
from .emotion import detect_emotion
from .tools import process_refund, escalate_ticket

MAX_TURNS = 3


class CustomerSupportEnv:
    def __init__(self, task_file="tasks/tasks.json"):
        with open(task_file) as f:
            self.tasks = json.load(f)

        self.state = None
        self.current_task = None
        self.turn = 0
        self.done = False

    def reset(self):
        self.current_task = random.choice(self.tasks)
        self.turn = 0
        self.done = False

        emotion = detect_emotion(self.current_task["query"])

        self.state = {
            "conversation": [
                {"role": "user", "content": self.current_task["query"]}
            ],
            "context": {
                "order_status": self.current_task.get("order_status", "unknown"),
                "history": []
            },
            "emotion": emotion
        }

        return Observation(**self.state)

    def step(self, action: Action):
        if self.done:
            raise Exception("Call reset()")

        self.turn += 1

        # Agent response
        self.state["conversation"].append({
            "role": "agent",
            "content": action.response
        })

        # ✅ update memory
        self.state["context"]["history"] = self.state["conversation"][-5:]

        # Tool usage
        tool_result = {}
        if action.decision == "refund":
            tool_result = process_refund("123")
        elif action.decision == "escalate":
            tool_result = escalate_ticket("issue")

        # Improved multi-turn follow-ups
        if self.turn < MAX_TURNS and action.decision == "ask_info":
            follow_ups = [
                "Here is my order ID: 123",
                "I already shared details earlier",
                "Why do you need that?"
            ]
            self.state["conversation"].append({
                "role": "user",
                "content": random.choice(follow_ups)
            })

        reward = grade_action(action, self.current_task, self.turn)

        if action.decision in ["refund", "resolve", "escalate"]:
            self.done = True
        elif self.turn >= MAX_TURNS:
            self.done = True

        return StepResult(
            observation=Observation(**self.state),
            reward=reward,
            done=self.done,
            info={"tool_result": tool_result, "turn": self.turn}
        )

    def state_fn(self):
        return self.state