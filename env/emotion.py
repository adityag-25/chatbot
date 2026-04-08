def detect_emotion(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["angry", "frustrated", "worst", "bad"]):
        return "angry"
    elif any(w in text for w in ["confused", "don't understand"]):
        return "confused"
    return "neutral"