import os
import logging
import requests
import datetime
import certifi
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pymongo import MongoClient
from retrying import retry

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Load environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "sentimentDB")

# Validate required environment variables
if not HF_API_KEY:
    logger.error("HF_API_KEY environment variable is not set")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is not set")

# Build headers for Hugging Face
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
} if HF_API_KEY else {}

# Enhanced retry wrapper for DB connection
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_db_connection():
    if not MONGO_URI:
        raise ConnectionError("MongoDB URI not configured")
    
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15000,
            socketTimeoutMS=30000,
            connectTimeoutMS=10000,
            tls=True,
            tlsCAFile=certifi.where(),
            retryWrites=True,
            appname="sentiment-app"
        )
        # Test connection
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        # More specific error handling
        if "SSL" in str(e) or "TLS" in str(e):
            logger.error("SSL/TLS issue detected. Check MongoDB Atlas network settings.")
        raise

def init_db():
    if not MONGO_URI:
        logger.warning("Skipping database initialization - MONGO_URI not configured")
        return
        
    try:
        client = get_db_connection()
        db = client[DB_NAME]
        
        # Check if database exists, create if not
        if DB_NAME not in client.list_database_names():
            logger.info(f"Creating database: {DB_NAME}")
        
        # Create collection only if it doesn't exist
        if "entries" not in db.list_collection_names():
            db.create_collection("entries")
            logger.info("Created 'entries' collection")
        
        client.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        # Don't crash the app if DB fails

# Initialize database
init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    if not MONGO_URI:
        return jsonify({
            "status": "degraded", 
            "database": "not_configured",
            "message": "MONGO_URI environment variable missing"
        }), 200
    
    try:
        client = get_db_connection()
        # Get some basic info to verify connection
        db = client[DB_NAME]
        count = db.entries.count_documents({})
        client.close()
        
        return jsonify({
            "status": "healthy", 
            "database": "connected",
            "entry_count": count
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy", 
            "database": "disconnected",
            "error": str(e)
        }), 500

@app.route("/debug", methods=["GET"])
def debug_env():
    # Mask sensitive information in the response
    masked_uri = "configured" if MONGO_URI else "not_configured"
    if MONGO_URI and "@" in MONGO_URI:
        masked_uri = MONGO_URI.split('@')[0].split('://')[0] + '://***:***@' + MONGO_URI.split('@')[1]
    
    return jsonify({
        "HF_API_KEY_configured": bool(HF_API_KEY),
        "MONGO_URI_configured": bool(MONGO_URI),
        "MONGO_URI_masked": masked_uri,
        "DB_NAME": DB_NAME,
        "environment": "production" if not os.getenv("FLASK_DEBUG") else "development"
    })

@app.route("/submit", methods=["POST"])
def submit_entry():
    try:
        # Get entry from form or JSON
        if request.is_json:
            data = request.get_json()
            entry = data.get("entry")
        else:
            entry = request.form.get("entry")

        if not entry:
            return jsonify({"error": "No entry provided"}), 400

        if not HF_API_KEY:
            return jsonify({"error": "Hugging Face API key is missing"}), 500

        # Hugging Face API request
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": entry})
        
        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), response.status_code

        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list):
            if isinstance(result[0], list):
                result = result[0]
        
        if not isinstance(result, list) or not all("label" in r and "score" in r for r in result):
            return jsonify({"error": f"Unexpected response format: {result}"}), 500

        dominant = max(result, key=lambda x: x['score'])
        label = dominant['label']
        score_value = dominant['score'] if label == 'POSITIVE' else -dominant['score']

        # Store in MongoDB if available
        db_success = False
        if MONGO_URI:
            try:
                client = get_db_connection()
                db = client[DB_NAME]
                db.entries.insert_one({
                    "entry": entry,
                    "timestamp": datetime.datetime.now(),
                    "label": label,
                    "score": score_value
                })
                client.close()
                db_success = True
                logger.info("Entry stored in MongoDB")
            except Exception as db_error:
                logger.error(f"Failed to store in MongoDB: {db_error}")
        else:
            logger.warning("MongoDB not configured - data not persisted")

        return jsonify({
            "result": result,
            "storage_status": "stored" if db_success else "not_stored"
        })

    except Exception as e:
        logger.error(f"Submission failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/entries", methods=["GET"])
def get_entries():
    if not MONGO_URI:
        return jsonify({"error": "MongoDB not configured"}), 400
        
    try:
        client = get_db_connection()
        db = client[DB_NAME]
        rows = list(db.entries.find({}, {"_id": 0, "timestamp": 1, "score": 1}).sort("timestamp", 1))

        labels = [r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for r in rows]
        scores = [r["score"] for r in rows]
        client.close()
        
        return jsonify({
            "labels": labels, 
            "scores": scores,
            "count": len(rows)
        })
    except Exception as e:
        logger.error(f"Failed to get entries: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
