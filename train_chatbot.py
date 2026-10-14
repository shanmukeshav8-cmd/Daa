import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
INTENTS_PATH = ROOT / "data" / "intents.json"
MODEL_PATH = ROOT / "models" / "chatbot_model.pkl"


def main():
    with INTENTS_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    examples = []
    labels = []
    for intent in data.get("intents", []):
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        if not tag or not patterns:
            raise ValueError("Every intent must have a tag and at least one pattern.")
        examples.extend(patterns)
        labels.extend([tag] * len(patterns))

    if len(set(labels)) < 2:
        raise ValueError("At least two distinct intent tags are required.")

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(examples, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained intent classifier with {len(examples)} examples across {len(set(labels))} intents.")
    print("Saved model to models/chatbot_model.pkl")


if __name__ == "__main__":
    main()
