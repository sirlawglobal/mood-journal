import os
import logging
import requests
import datetime
import certifi
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pymongo import MongoClient
from retrying import retry

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

# MongoDB connection string - use the exact URI from Render environment variable
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is not set")
    
DB_NAME = os.getenv("DB_NAME", "sentimentDB")

# Build headers for Hugging Face
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
} if HF_API_KEY else {}

# Retry wrapper for DB connection
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_db_connection():
    try:
        # For Render deployment, use the connection string directly
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10000,  # Increased timeout
            socketTimeoutMS=30000,
            connectTimeoutMS=10000,
            tls=True,
            tlsCAFile=certifi.where(),  # ensure SSL certs are trusted
            retryWrites=True,
            appname="sentiment-app"
        )
        # Test connection
        client.admin.command("ping")
        logger.debug("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise

def init_db():
    try:
        client = get_db_connection()
        db = client[DB_NAME]
        # Create collection only if it doesn't exist
        if "entries" not in db.list_collection_names():
            db.create_collection("entries")
        logger.debug("Database and collection 'entries' initialized successfully")
        client.close()
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        # Don't raise exception here to allow app to start without DB

# Initialize database (but don't crash if it fails)
try:
    if MONGO_URI:
        init_db()
    else:
        logger.warning("Skipping DB initialization - MONGO_URI not set")
except Exception as e:
    logger.error(f"Failed to initialize database on startup: {str(e)}")
    # Continue without DB connection

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    try:
        if not MONGO_URI:
            return jsonify({"status": "unhealthy", "error": "MONGO_URI not configured"}), 500
            
        client = get_db_connection()
        client.close()
        return jsonify({"status": "healthy", "database": "connected"})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug_env():
    return jsonify({
        "HF_API_KEY": bool(HF_API_KEY),
        "MONGO_URI_configured": bool(MONGO_URI),
        "DB_NAME": DB_NAME
    })

@app.route("/submit", methods=["POST"])
def submit_entry():
    logger.debug("Submit entry route hit")
    try:
        entry = request.form.get("entry")
        if not entry and request.is_json:
            entry = request.json.get("entry")

        if not entry:
            return jsonify({"error": "No entry provided"}), 400

        logger.debug(f"Received entry: {entry}")

        if not HF_API_KEY:
            return jsonify({"error": "Hugging Face API key is missing"}), 500

        # Hugging Face API request
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": entry})
        logger.debug(f"Hugging Face response: {response.status_code}, {response.text}")

        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), response.status_code

        result = response.json()
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        if not isinstance(result, list) or not all("label" in r and "score" in r for r in result):
            raise ValueError(f"Unexpected response format: {result}")

        dominant = max(result, key=lambda x: x['score'])
        label = dominant['label']
        score_value = dominant['score'] if label == 'POSITIVE' else -dominant['score']

        timestamp = datetime.datetime.now()

        # Insert into MongoDB if connection is available
        if MONGO_URI:
            try:
                client = get_db_connection()
                db = client[DB_NAME]
                db.entries.insert_one({
                    "entry": entry,
                    "timestamp": timestamp,
                    "label": label,
                    "score": score_value
                })
                client.close()
                logger.debug("Entry inserted into MongoDB")
            except Exception as db_error:
                logger.error(f"Failed to insert into MongoDB: {db_error}")
                # Continue without storing in DB
        else:
            logger.warning("MongoDB not configured - skipping data storage")

        return jsonify({"result": result})

    except Exception as e:
        logger.error(f"Submission failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/entries", methods=["GET"])
def get_entries():
    try:
        if not MONGO_URI:
            return jsonify({"error": "MongoDB not configured"}), 500
            
        client = get_db_connection()
        db = client[DB_NAME]
        rows = list(db.entries.find({}, {"_id": 0, "timestamp": 1, "score": 1}).sort("timestamp", 1))

        labels = [r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for r in rows]
        scores = [r["score"] for r in rows]
        logger.debug(f"Fetched {len(rows)} entries from MongoDB")
        client.close()
        return jsonify({"labels": labels, "scores": scores})
    except Exception as e:
        logger.error(f"Failed to get entries: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Don't run in debug mode on Render
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
