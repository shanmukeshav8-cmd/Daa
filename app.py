import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
INTENTS_PATH = ROOT / "data" / "intents.json"
MODEL_PATHS = {
    "chatbot": MODEL_DIR / "chatbot_model.pkl",
    "delay": MODEL_DIR / "delay_model.pkl",
    "queue": MODEL_DIR / "queue_model.pkl",
}
DELAY_FIELDS = [
    "airline",
    "origin",
    "destination",
    "scheduled_hour",
    "day_of_week",
    "weather",
    "traffic_level",
]
QUEUE_FIELDS = ["terminal", "hour", "day_of_week", "passenger_volume", "open_lanes"]

app = Flask(__name__)


@lru_cache(maxsize=3)
def load_model(name):
    path = MODEL_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Model '{name}' is not trained. Run train_chatbot.py, train_delay.py, and train_queue.py."
        )
    return joblib.load(path)


def model_error(error):
    return jsonify({"error": str(error)}), 503


def read_payload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Request body must be a JSON object."}), 400)
    return payload, None


def require_text(payload, field, max_length=80):
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > max_length:
        raise ValueError(f"'{field}' must be a non-empty string of at most {max_length} characters.")
    return value.strip()


def require_integer(payload, field, minimum, maximum):
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"'{field}' must be an integer between {minimum} and {maximum}.")
    return value


def require_choice(payload, field, choices):
    value = require_text(payload, field)
    if value not in choices:
        raise ValueError(f"'{field}' must be one of: {', '.join(choices)}.")
    return value


def check_fields(payload, fields):
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}.")


def render_home(**context):
    context["models_ready"] = all(path.is_file() for path in MODEL_PATHS.values())
    return render_template("index.html", **context)


def form_payload(integer_fields):
    payload = request.form.to_dict()
    for field in integer_fields:
        try:
            payload[field] = int(payload[field])
        except (KeyError, ValueError):
            pass
    return payload


def chat_response(message):
    message = require_text({"message": message}, "message", max_length=500)
    model = load_model("chatbot")
    prediction = model.predict([message])[0]
    confidence = float(max(model.predict_proba([message])[0]))

    with INTENTS_PATH.open(encoding="utf-8") as file:
        intents = json.load(file)["intents"]
    selected = next((intent for intent in intents if intent["tag"] == prediction), None)
    if selected is None:
        raise RuntimeError("The predicted intent has no configured response.")
    if confidence < 0.20:
        selected = next((intent for intent in intents if intent["tag"] == "fallback"), None)
        if selected is None:
            raise RuntimeError("The intent data is missing a fallback response.")

    return {
        "reply": selected["responses"][0],
        "intent": selected["tag"],
        "confidence": round(confidence, 3),
    }


def delay_estimate(payload):
    check_fields(payload, DELAY_FIELDS)
    features = {
        "airline": require_choice(payload, "airline", ["Aurora Air", "Meridian Air", "Northstar", "Pacific Connect"]),
        "origin": require_choice(payload, "origin", ["AAA", "BBB", "CCC"]).upper(),
        "destination": require_choice(payload, "destination", ["AAA", "BBB", "CCC"]).upper(),
        "scheduled_hour": require_integer(payload, "scheduled_hour", 0, 23),
        "day_of_week": require_integer(payload, "day_of_week", 0, 6),
        "weather": require_choice(payload, "weather", ["clear", "light_rain", "fog", "storm"]),
        "traffic_level": require_choice(payload, "traffic_level", ["low", "medium", "high"]),
    }
    model = load_model("delay")
    minutes = max(0, float(model.predict(pd.DataFrame([features]))[0]))
    return {
        "estimated_delay_minutes": round(minutes),
        "notice": "Estimate from sample data; not live flight status.",
    }


def queue_estimate(payload):
    check_fields(payload, QUEUE_FIELDS)
    features = {
        "terminal": require_integer(payload, "terminal", 1, 3),
        "hour": require_integer(payload, "hour", 0, 23),
        "day_of_week": require_integer(payload, "day_of_week", 0, 6),
        "passenger_volume": require_integer(payload, "passenger_volume", 20, 300),
        "open_lanes": require_integer(payload, "open_lanes", 1, 16),
    }
    model = load_model("queue")
    minutes = max(0, float(model.predict(pd.DataFrame([features]))[0]))
    return {
        "estimated_wait_minutes": round(minutes),
        "notice": "Estimate from sample data; not live queue conditions.",
    }


@app.get("/")
def index():
    return render_home()


@app.post("/chat")
def web_chat():
    message = request.form.get("message", "")
    try:
        result = chat_response(message)
    except (FileNotFoundError, ValueError) as exc:
        return render_home(chat_message=message, chat_error=str(exc)), 400 if isinstance(exc, ValueError) else 503
    except RuntimeError as exc:
        return render_home(chat_message=message, chat_error=str(exc)), 500
    return render_home(chat_message=message, chat_reply=result["reply"])


@app.post("/predict/delay")
def web_predict_delay():
    payload = form_payload(["scheduled_hour", "day_of_week"])
    try:
        result = delay_estimate(payload)
    except (FileNotFoundError, ValueError) as exc:
        return render_home(delay_error=str(exc)), 400 if isinstance(exc, ValueError) else 503
    return render_home(delay_result=result)


@app.post("/predict/queue")
def web_predict_queue():
    payload = form_payload(["terminal", "hour", "day_of_week", "passenger_volume", "open_lanes"])
    try:
        result = queue_estimate(payload)
    except (FileNotFoundError, ValueError) as exc:
        return render_home(queue_error=str(exc)), 400 if isinstance(exc, ValueError) else 503
    return render_home(queue_result=result)


@app.get("/api/health")
def health():
    status = {name: path.is_file() for name, path in MODEL_PATHS.items()}
    return jsonify({"status": "ready" if all(status.values()) else "models_missing", "models": status})


@app.post("/api/chat")
def chat():
    payload, error = read_payload()
    if error:
        return error
    try:
        result = chat_response(payload.get("message"))
    except FileNotFoundError as exc:
        return model_error(exc)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/predict/delay")
def predict_delay():
    payload, error = read_payload()
    if error:
        return error
    try:
        result = delay_estimate(payload)
    except FileNotFoundError as exc:
        return model_error(exc)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.post("/api/predict/queue")
def predict_queue():
    payload, error = read_payload()
    if error:
        return error
    try:
        result = queue_estimate(payload)
    except FileNotFoundError as exc:
        return model_error(exc)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
