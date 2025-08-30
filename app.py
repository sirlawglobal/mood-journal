import os
import logging
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Hugging Face API settings
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
HF_API_KEY = os.getenv("HF_API_KEY")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit_entry():
    logger.debug("Submit entry route hit")

    try:
        entry = request.form.get("entry")
        if not entry:
            return jsonify({"error": "No entry provided"}), 400

        logger.debug(f"Received entry: {entry}")

        # Send request to Hugging Face
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": entry})
        logger.debug(f"Hugging Face response: {response.status_code}, {response.text}")

        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), 500

        result = response.json()

        # Handle response format safely
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        if not isinstance(result, list) or not all("label" in r and "score" in r for r in result):
            raise ValueError(f"Unexpected response format: {result}")

        return jsonify({"result": result})

    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render requires dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
