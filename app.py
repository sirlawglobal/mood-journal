import os
import logging
import requests
import datetime
import certifi
import socket
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

# Convert SRV URI to direct connection if needed
def get_mongo_connection_string():
    if not MONGO_URI:
        return None
    
    # If it's already a direct connection, return as is
    if MONGO_URI.startswith('mongodb://'):
        return MONGO_URI
    
    # If it's SRV connection, use it directly (pymongo handles SRV properly)
    if MONGO_URI.startswith('mongodb+srv://'):
        return MONGO_URI
    
    return None

# Simple connection function for Render
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_db_connection():
    if not MONGO_URI:
        raise ConnectionError("MongoDB URI not configured")
    
    try:
        # Use the original SRV connection string - pymongo handles SRV resolution
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=20000,
            socketTimeoutMS=30000,
            connectTimeoutMS=15000,
            tls=True,
            tlsCAFile=certifi.where(),
            retryWrites=True,
            appname="sentiment-app-render",
            connect=False  # Don't connect immediately
        )
        
        # Force connection and test
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
        
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        
        # If SRV fails, try direct connection with specific hosts
        try:
            logger.info("Trying direct connection...")
            # Extract components from SRV URI
            if MONGO_URI.startswith('mongodb+srv://'):
                # This is the format MongoDB Atlas provides
                # Use the exact replica set members from your error message
                direct_uri = "mongodb://akanjilawrence9999_db_user:qRRt2NE0TaTuk6pE@ac-lwq5ipy-shard-00-00.dpyyhxs.mongodb.net:27017,ac-lwq5ipy-shard-00-01.dpyyhxs.mongodb.net:27017,ac-lwq5ipy-shard-00-02.dpyyhxs.mongodb.net:27017/sentimentDB?ssl=true&replicaSet=atlas-14b6h0-shard-0&authSource=admin&retryWrites=true&w=majority"
                
                client = MongoClient(
                    direct_uri,
                    serverSelectionTimeoutMS=20000,
                    socketTimeoutMS=30000,
                    connectTimeoutMS=15000,
                    tls=True,
                    tlsCAFile=certifi.where(),
                    retryWrites=True,
                    appname="sentiment-app-direct"
                )
                client.admin.command('ping')
                logger.info("Successfully connected using direct connection")
                return client
                
        except Exception as direct_error:
            logger.error(f"Direct connection also failed: {direct_error}")
            raise ConnectionError(f"All connection attempts failed: {e}, {direct_error}")

def init_db():
    if not MONGO_URI:
        logger.warning("Skipping database initialization - MONGO_URI not configured")
        return
        
    try:
        client = get_db_connection()
        db = client[DB_NAME]
        
        if "entries" not in db.list_collection_names():
            db.create_collection("entries")
            logger.info("Created 'entries' collection")
        
        client.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

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
            "database": "not_configured"
        }), 200
    
    try:
        client = get_db_connection()
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
            "error": "Check MongoDB connection"
        }), 500

@app.route("/debug", methods=["GET"])
def debug_env():
    return jsonify({
        "HF_API_KEY_configured": bool(HF_API_KEY),
        "MONGO_URI_configured": bool(MONGO_URI),
        "DB_NAME": DB_NAME
    })

@app.route("/submit", methods=["POST"])
def submit_entry():
    try:
        entry = request.form.get("entry") or (request.json.get("entry") if request.is_json else None)
        if not entry:
            return jsonify({"error": "No entry provided"}), 400

        if not HF_API_KEY:
            return jsonify({"error": "Hugging Face API key is missing"}), 500

        # Hugging Face API request
        response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": entry})
        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), response.status_code

        result = response.json()
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]

        dominant = max(result, key=lambda x: x['score'])
        label = dominant['label']
        score_value = dominant['score'] if label == 'POSITIVE' else -dominant['score']

        # Try to store in MongoDB
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
            except Exception as db_error:
                logger.error(f"MongoDB storage failed: {db_error}")

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
        
        return jsonify({"labels": labels, "scores": scores, "count": len(rows)})
    except Exception as e:
        logger.error(f"Failed to get entries: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Simple test endpoint
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Flask app is running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
