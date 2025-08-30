import os
import logging
import requests
import datetime
import mysql.connector
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
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

# MySQL database settings from environment variables
DB_HOST = os.getenv("DB_HOST", "mysq-sirlawdev-zenkonect.d.aivencloud.com")
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "defaultdb")
DB_PORT = int(os.getenv("DB_PORT", 26754))
DB_SSL_CA = os.getenv("DB_SSL_CA", "ca.pem")  # Path to CA certificate

# Build headers for Hugging Face
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
} if HF_API_KEY else {}

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_db_connection(use_db=True):
    try:
        config = {
            "host": DB_HOST,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "port": DB_PORT,
            "ssl_ca": DB_SSL_CA,
            "ssl_verify_cert": True
        }
        if use_db:
            config["database"] = DB_NAME
        conn = mysql.connector.connect(**config)
        logger.debug("Successfully connected to MySQL")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
        raise

def init_db():
    try:
        # Connect without database to create it if needed
        conn = get_db_connection(use_db=False)
        c = conn.cursor()
        c.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.commit()
        conn.close()

        # Connect to the database and create table
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
        logger.debug("Database and table 'entries' initialized successfully")
        conn.close()
    except mysql.connector.Error as err:
        logger.error(f"Failed to initialize database: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
        raise

# Initialize database
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database on startup: {str(e)}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"status": "healthy", "database": "connected"})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug_env():
    return jsonify({
        "HF_API_KEY": bool(HF_API_KEY),
        "DB_HOST": DB_HOST,
        "DB_USER": DB_USER,
        "DB_PASSWORD": "****" if DB_PASSWORD else None,
        "DB_NAME": DB_NAME,
        "DB_PORT": DB_PORT,
        "DB_SSL_CA": bool(DB_SSL_CA)
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
            return jsonify({"error": "Hugging Face API key is missing. Set HF_API_KEY in environment variables"}), 500

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
        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO entries (entry, timestamp, label, score) VALUES (%s, %s, %s, %s)",
                (entry, timestamp, label, score_value)
            )
            conn.commit()
            logger.debug("Entry inserted into database")
        except mysql.connector.Error as err:
            logger.error(f"Database insert failed: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
            return jsonify({"error": f"Database error: {err.msg}"}), 500
        finally:
            conn.close()

        return jsonify({"result": result})

    except mysql.connector.Error as err:
        logger.error(f"Submission failed: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
        return jsonify({"error": f"Database error: {err.msg}"}), 500
    except Exception as e:
        logger.error(f"Submission failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/entries", methods=["GET"])
def get_entries():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute("SELECT timestamp, score FROM entries ORDER BY timestamp ASC")
            rows = c.fetchall()
            labels = [row[0].strftime("%Y-%m-%d %H:%M:%S") for row in rows]
            scores = [row[1] for row in rows]
            logger.debug(f"Fetched {len(rows)} entries from database")
            return jsonify({"labels": labels, "scores": scores})
        except mysql.connector.Error as err:
            logger.error(f"Database query failed: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
            return jsonify({"error": f"Database error: {err.msg}"}), 500
        finally:
            conn.close()
    except mysql.connector.Error as err:
        logger.error(f"Failed to get entries: errno={err.errno}, msg={err.msg}, sqlstate={err.sqlstate}")
        return jsonify({"error": f"Database error: {err.msg}"}), 500
    except Exception as e:
        logger.error(f"Failed to get entries: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
