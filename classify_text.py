from __future__ import annotations

import json
import sys

from pdf_ai_classifier import load_model


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("Enter service/manual text to classify: ").strip()

    model = load_model()
    category, confidence = model.predict(text)
    probabilities = model.predict_proba(text)

    print(json.dumps({
        "input": text,
        "prediction": category,
        "confidence": round(confidence, 4),
        "top_probabilities": {k: round(v, 4) for k, v in list(probabilities.items())[:5]},
    }, indent=2))


if __name__ == "__main__":
    main()
