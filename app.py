from flask import Flask, request, jsonify, render_template
import requests
import mysql.connector
import time
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging (debug level so everything shows in Render logs)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get environment variables
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    logger.error("HF_API_TOKEN not set in environment variables")
    raise ValueError("HF_API_TOKEN environment variable not set")

# Hugging Face Sentiment API
# HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

# Hugging Face Sentiment API
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"


# MySQL Config (Render or local DB)
db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "mood_journal"),
}

# Database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        logger.debug("Database connection successful")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        raise

# Insert new entry
def insert_entry(entry_text, sentiment, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO entries (entry_text, sentiment, score) VALUES (%s, %s, %s)"
        cursor.execute(query, (entry_text, sentiment, score))
        conn.commit()
        logger.debug("Entry inserted into DB successfully")
    except mysql.connector.Error as err:
        logger.error(f"Insert failed: {err}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

# Fetch all entries
def get_all_entries():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT * FROM entries ORDER BY created_at ASC"
        cursor.execute(query)
        rows = cursor.fetchall()
        logger.debug(f"Fetched {len(rows)} entries from DB")
        return rows
    except mysql.connector.Error as err:
        logger.error(f"Read failed: {err}")
        raise
    finally:
        cursor.close()
        conn.close()

# Sentiment analysis using Hugging Face
def analyze_sentiment(text, retries=3, delay=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    for attempt in range(retries):
        try:
            logger.debug(f"Sending request to Hugging Face (attempt {attempt+1}) with text: {text}")
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)

            try:
                result = response.json()
            except Exception:
                result = response.text

            logger.debug(f"Hugging Face response: {response.status_code}, {result}")

            if response.status_code == 200:
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    label = result[0].get("label")
                    score = result[0].get("score")
                    emotion = "happy" if label and label.upper() == "POSITIVE" else "sad"
                    return emotion, score
                else:
                    raise ValueError(f"Unexpected response format: {result}")
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, retrying...")
                time.sleep(delay)
                continue
            else:
                raise Exception(f"API error: {result}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise

    raise Exception("API request failed after retries")

# Flask Routes
@app.route("/")
def index():
    logger.debug("Index route hit")
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit_entry():
    logger.debug("Submit entry route hit")
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    entry_text = data.get("entry")
    logger.debug(f"Received entry: {entry_text}")

    if not entry_text:
        return jsonify({"error": "No entry provided"}), 400

    try:
        emotion, score = analyze_sentiment(entry_text)
        insert_entry(entry_text, emotion, score)
        return jsonify({"message": "Entry saved", "emotion": emotion, "score": score})
    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/entries", methods=["GET"])
def fetch_entries():
    logger.debug("Fetch entries route hit")
    try:
        entries = get_all_entries()
        chart_data = {
            "labels": [entry["created_at"].strftime("%Y-%m-%d %H:%M") for entry in entries],
            "scores": [entry["score"] if entry["sentiment"] == "happy" else -entry["score"] for entry in entries],
        }
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Fetch entries failed: {e}")
        return jsonify({"error": "Failed to fetch entries"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
