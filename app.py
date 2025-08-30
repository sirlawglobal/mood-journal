import os
import logging
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# ✅ Use the correct Hugging Face model endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit_entry():
    logger.debug("Submit entry route hit")
    data = request.get_json()
    entry_text = data.get("entry", "").strip()
    logger.debug(f"Received entry: {entry_text}")

    if not entry_text:
        return jsonify({"error": "Entry cannot be empty"}), 400

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": entry_text}

        logger.debug(f"Sending request to Hugging Face with text: {entry_text}")
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        logger.debug(f"Hugging Face response: {response.status_code}, {response.text}")

        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")

        response_json = response.json()

        # ✅ Handle both response formats
        if isinstance(response_json, list) and len(response_json) > 0:
            if isinstance(response_json[0], list):  # nested list
                predictions = response_json[0]
            else:
                predictions = response_json
        else:
            raise ValueError(f"Unexpected response format: {response_json}")

        # Pick the label with highest score
        result = max(predictions, key=lambda x: x["score"])
        sentiment = result["label"]
        confidence = result["score"]

        logger.debug(f"Final sentiment: {sentiment} ({confidence:.2f})")
        return jsonify({"sentiment": sentiment, "confidence": confidence})

    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Debug mode for local dev
    app.run(debug=True)
