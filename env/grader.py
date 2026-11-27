"""
Reward / grading module for CustomerSupportEnv.

Rubric (matches README exactly):
  Intent correctness   : 25%
  Decision accuracy    : 35%
  Response quality     : 25%  (keyword overlap + light semantic heuristic)
  Emotion handling     : 10%
  Efficiency           : 5%
"""

from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_overlap(response: str, keywords: List[str]) -> float:
    """
    Fraction of the task's keyword list that appears in the response.
    Returns 0.0–1.0.
    """
    if not keywords:
        return 0.0
    lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def _semantic_heuristic(response: str, ideal: str) -> float:
    """
    Light semantic similarity without external models.
    Computes word-level Jaccard similarity between response and ideal_response.
    """
    if not ideal:
        return 0.5  # neutral when no ideal provided

    r_words = set(response.lower().split())
    i_words = set(ideal.lower().split())
    if not r_words or not i_words:
        return 0.0

    intersection = r_words & i_words
    union = r_words | i_words
    jaccard = len(intersection) / len(union)

    # Scale: even a low Jaccard can earn partial credit
    # Clamp between 0.2 and 1.0 so there's always a floor for a reasonable response
    return min(1.0, max(0.2, jaccard * 2.5))


def _response_quality(response: str, task: dict) -> float:
    """
    Combined response quality score (0.0–1.0).
    Blends keyword overlap (60%) and semantic heuristic (40%).
    """
    keywords: List[str] = task.get("keywords", [])
    ideal: str = task.get("ideal_response", "")

    kw_score = _keyword_overlap(response, keywords)
    sem_score = _semantic_heuristic(response, ideal)

    return 0.6 * kw_score + 0.4 * sem_score


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

def grade_action(action, task: dict, turn: int) -> float:
    """
    Compute a 0.0–1.0 reward for a single agent action.

    Components
    ----------
    intent_score   : 0.25  — exact match against task["intent"]
    decision_score : 0.35  — exact match against task["decision"]
                             partial credit (0.10) for a "close" decision
    quality_score  : 0.25  — keyword overlap + semantic heuristic
    emotion_score  : 0.10  — angry tasks require an apology
    efficiency     : 0.05  — decays slightly with each extra turn
    """
    score = 0.0

    # --- Intent (25%) -------------------------------------------------------
    if action.intent == task.get("intent", ""):
        score += 0.25

    # --- Decision (35%) -----------------------------------------------------
    correct_decision = task.get("decision", "")
    if action.decision == correct_decision:
        score += 0.35
    else:
        # Partial credit for related decisions (e.g. resolve ↔ ask_info)
        related_pairs = {
            frozenset({"resolve", "ask_info"}),
            frozenset({"escalate", "refund"}),
        }
        pair = frozenset({action.decision, correct_decision})
        if pair in related_pairs:
            score += 0.10
        # No penalty — the original -0.2 made scores go negative too easily

    # --- Response Quality (25%) ---------------------------------------------
    score += 0.25 * _response_quality(action.response, task)

    # --- Emotion Handling (10%) ---------------------------------------------
    task_emotion = task.get("emotion", "neutral")
    if task_emotion == "angry":
        apology_words = ["sorry", "apologize", "apologies", "regret"]
        if any(w in action.response.lower() for w in apology_words):
            score += 0.10
        else:
            score += 0.03  # partial — at least attempt to handle it
    elif task_emotion == "confused":
        clarify_words = ["please", "share", "provide", "detail", "help", "clarify"]
        if any(w in action.response.lower() for w in clarify_words):
            score += 0.10
        else:
            score += 0.03
    else:
        # Neutral — always full marks for emotion handling
        score += 0.10

    # --- Efficiency (5%) ----------------------------------------------------
    # Full 0.05 on turn 1, 0.03 on turn 2, 0.01 on turn 3, 0 beyond
    efficiency = max(0.0, 0.05 - (turn - 1) * 0.02)
    score += efficiency

    return round(max(0.0, min(1.0, score)), 4)
