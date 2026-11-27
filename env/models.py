from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Message(BaseModel):
    role: str
    content: str


class Observation(BaseModel):
    conversation: List[Message]
    context: Dict[str, Any]
    emotion: str


class Action(BaseModel):
    intent: str
    decision: str
    response: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
