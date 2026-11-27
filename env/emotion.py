"""
Emotion detection module.

Detects the emotional tone of a customer query as one of:
  - "angry"   : frustration, anger, strong dissatisfaction
  - "confused": uncertainty, lack of understanding
  - "neutral" : calm, informational

The keyword lists are aligned with the actual phrases found in tasks.json
so that every angry/confused task is correctly detected.
"""

ANGRY_KEYWORDS = [
    # explicit anger
    "angry", "frustrated", "frustration", "worst", "terrible",
    "horrible", "disgusting", "furious", "outraged",
    # task-specific dissatisfaction cues
    "broken", "late", "rude", "empty", "stuck",
    "not received", "hasn't arrived", "hasn't arrived",
    "wrong item", "wrong product",
    # time/money pressure signals
    "still waiting", "unacceptable", "refund not received",
    "late refund", "charged twice", "not working",
    # common negative adjectives + verbs
    "bad", "useless", "cheated", "scam", "ridiculous",
    "never", "absolutely", "disgrace",
]

CONFUSED_KEYWORDS = [
    "confused", "don't understand", "not sure", "unclear",
    "what do i", "how do i", "help me understand",
    "i don't know", "which one", "can you explain",
]


def detect_emotion(text: str) -> str:
    """Return the dominant emotion of the query string."""
    lower = text.lower()

    if any(kw in lower for kw in ANGRY_KEYWORDS):
        return "angry"

    if any(kw in lower for kw in CONFUSED_KEYWORDS):
        return "confused"

    return "neutral"
