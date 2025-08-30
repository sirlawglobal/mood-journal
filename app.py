import os
import logging
import requests
import datetime
import mysql.connector
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

# MySQL database settings from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))  # Default to 3306 if not set

# Build headers safely for Hugging Face
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
} if HF_API_KEY else {}

# Function to get database connection
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

# Initialize database table
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            entry TEXT,
            timestamp DATETIME,
            label VARCHAR(255),
            score FLOAT
        )
    ''')
    conn.commit()
    conn.close()

# Call init_db when the app starts
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit_entry():
    logger.debug("Submit entry route hit")

    try:
        # Get entry from form or JSON
        entry = request.form.get("entry")
        if not entry and request.is_json:
            entry = request.json.get("entry")

        if not entry:
            return jsonify({"error": "No entry provided"}), 400

        logger.debug(f"Received entry: {entry}")

        # Ensure API key is present
        if not HF_API_KEY:
            return jsonify({"error": "Hugging Face API key is missing. Set HF_API_KEY in .env"}), 500

        # Send request to Hugging Face
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": entry})
        logger.debug(f"Hugging Face response: {response.status_code}, {response.text}")

        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), response.status_code

        result = response.json()

        # Handle response format safely
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        if not isinstance(result, list) or not all("label" in r and "score" in r for r in result):
            raise ValueError(f"Unexpected response format: {result}")

        # Determine dominant sentiment and signed score
        dominant = max(result, key=lambda x: x['score'])
        label = dominant['label']
        score_value = dominant['score'] if label == 'POSITIVE' else -dominant['score']

        # Store in database
        timestamp = datetime.datetime.now()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO entries (entry, timestamp, label, score) VALUES (%s, %s, %s, %s)",
            (entry, timestamp, label, score_value)
        )
        conn.commit()
        conn.close()

        return jsonify({"result": result})

    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/entries", methods=["GET"])
def get_entries():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT timestamp, score FROM entries ORDER BY timestamp ASC")
        rows = c.fetchall()
        conn.close()

        labels = [row[0].strftime("%Y-%m-%d %H:%M:%S") for row in rows]
        scores = [row[1] for row in rows]

        return jsonify({"labels": labels, "scores": scores})
    except Exception as e:
        logger.error(f"Failed to get entries: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render requires dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
