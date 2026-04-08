def semantic_score(response, ideal):
    # Mock semantic similarity
    if ideal.lower() in response.lower():
        return 1.0
    return 0.5


def grade_action(action, task, turn):
    score = 0.0

    # Intent
    if action.intent == task["intent"]:
        score += 0.25

    # Decision
    if action.decision == task["decision"]:
        score += 0.35
    else:
        score -= 0.2

    # Semantic scoring (mock version)
    score += 0.5 * semantic_score(
        action.response,
        task.get("ideal_response", "")
    )

    # Emotion bonus
    if task.get("emotion") == "angry" and "sorry" in action.response.lower():
        score += 0.1

    # Efficiency
    score += max(0, 0.05 - turn * 0.01)

    return max(0.0, min(1.0, score))