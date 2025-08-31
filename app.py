import os
import logging
import requests
import datetime
import certifi
import ssl
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

# Multiple connection strategies for SSL issues
def get_db_connection():
    if not MONGO_URI:
        raise ConnectionError("MongoDB URI not configured")
    
    connection_methods = [
        try_connection_with_certifi,
        try_connection_with_ssl_context,
        try_connection_without_ssl_validation,
        try_connection_direct
    ]
    
    for method in connection_methods:
        try:
            client = method()
            logger.info(f"Successfully connected using {method.__name__}")
            return client
        except Exception as e:
            logger.warning(f"Connection method {method.__name__} failed: {e}")
            continue
    
    raise ConnectionError("All connection methods failed")

def try_connection_with_certifi():
    """Standard connection with certifi"""
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=20000,
        socketTimeoutMS=30000,
        connectTimeoutMS=15000,
        tls=True,
        tlsCAFile=certifi.where(),
        retryWrites=True,
        appname="sentiment-app-certifi"
    )
    client.admin.command("ping")
    return client

def try_connection_with_ssl_context():
    """Try with custom SSL context"""
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=20000,
        socketTimeoutMS=30000,
        connectTimeoutMS=15000,
        tls=True,
        tlsAllowInvalidCertificates=True,
        retryWrites=True,
        appname="sentiment-app-ssl-context"
    )
    client.admin.command("ping")
    return client

def try_connection_without_ssl_validation():
    """Try without SSL validation (for testing)"""
    # Create a modified URI without SSL
    if 'mongodb+srv://' in MONGO_URI:
        # For SRV connection, we need to use the direct connection string
        # Extract the base part and convert to direct connection
        base_uri = MONGO_URI.split('@')[1] if '@' in MONGO_URI else MONGO_URI
        direct_uri = f"mongodb://{MONGO_URI.split('://')[1].split('@')[0]}@{base_uri}"
        direct_uri = direct_uri.replace('?retryWrites=true&w=majority', '?retryWrites=true&w=majority&ssl=false&directConnection=true')
    else:
        direct_uri = MONGO_URI + '&ssl=false'
    
    client = MongoClient(
        direct_uri,
        serverSelectionTimeoutMS=20000,
        socketTimeoutMS=30000,
        connectTimeoutMS=15000,
        retryWrites=True,
        appname="sentiment-app-no-ssl"
    )
    client.admin.command("ping")
    return client

def try_connection_direct():
    """Try direct connection with specific hosts"""
    # Extract credentials from URI
    if '@' in MONGO_URI:
        auth_part = MONGO_URI.split('://')[1].split('@')[0]
        hosts_part = MONGO_URI.split('@')[1].split('/')[0]
        db_part = MONGO_URI.split('/')[3].split('?')[0]
        
        direct_uri = f"mongodb://{auth_part}@{hosts_part}/{db_part}?retryWrites=true&w=majority&ssl=true&directConnection=false"
    else:
        direct_uri = MONGO_URI
    
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
    client.admin.command("ping")
    return client

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_db_connection_retry():
    return get_db_connection()

def init_db():
    if not MONGO_URI:
        logger.warning("Skipping database initialization - MONGO_URI not configured")
        return
        
    try:
        client = get_db_connection_retry()
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
        client = get_db_connection_retry()
        db_stats = client.admin.command("dbstats")
        client.close()
        
        return jsonify({
            "status": "healthy", 
            "database": "connected"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy", 
            "database": "disconnected",
            "error": "Check MongoDB Atlas network settings and SSL configuration"
        }), 500

@app.route("/test-connection", methods=["GET"])
def test_connection():
    """Test endpoint to diagnose connection issues"""
    if not MONGO_URI:
        return jsonify({"error": "MONGO_URI not configured"}), 400
    
    results = {}
    methods = [
        ("certifi", try_connection_with_certifi),
        ("ssl_context", try_connection_with_ssl_context),
        ("no_ssl_validation", try_connection_without_ssl_validation),
        ("direct", try_connection_direct)
    ]
    
    for name, method in methods:
        try:
            client = method()
            client.admin.command("ping")
            client.close()
            results[name] = "success"
        except Exception as e:
            results[name] = f"failed: {str(e)}"
    
    return jsonify({"connection_tests": results})

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
                client = get_db_connection_retry()
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
        client = get_db_connection_retry()
        db = client[DB_NAME]
        rows = list(db.entries.find({}, {"_id": 0, "timestamp": 1, "score": 1}).sort("timestamp", 1))

        labels = [r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") for r in rows]
        scores = [r["score"] for r in rows]
        client.close()
        
        return jsonify({"labels": labels, "scores": scores, "count": len(rows)})
    except Exception as e:
        logger.error(f"Failed to get entries: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
