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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get environment variables (for Render or local use)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN environment variable not set")

# MySQL configuration (update with Render database credentials or external MySQL)
db_config = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", ""),
    'database': os.getenv("DB_NAME", "mood_journal")
}

# Hugging Face sentiment model
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        raise

# CRUD: Create - Insert new entry
def insert_entry(entry_text, sentiment, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO entries (entry_text, sentiment, score) VALUES (%s, %s, %s)"
        cursor.execute(query, (entry_text, sentiment, score))
        conn.commit()
    except mysql.connector.Error as err:
        logger.error(f"Insert failed: {err}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

# CRUD: Read - Get all entries for charting
def get_all_entries():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT * FROM entries ORDER BY created_at ASC"
        cursor.execute(query)
        return cursor.fetchall()
    except mysql.connector.Error as err:
        logger.error(f"Read failed: {err}")
        raise
    finally:
        cursor.close()
        conn.close()

# Analyze sentiment using Hugging Face API (fixed parser)
def analyze_sentiment(text, retries=3, delay=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    for attempt in range(retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
            result = response.json() if response.status_code == 200 else response.text
            logger.debug(f"Hugging Face response (attempt {attempt+1}): Status {response.status_code}, Data {result}")

            if response.status_code == 200:
                if isinstance(result, list) and len(result) > 0:
                    label = result[0]['label']
                    score = result[0]['score']
                    emotion = 'happy' if label.upper() == 'POSITIVE' else 'sad'
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_entry():
    data = request.json
    entry_text = data.get('entry')
    if not entry_text:
        return jsonify({'error': 'No entry provided'}), 400

    try:
        emotion, score = analyze_sentiment(entry_text)
        insert_entry(entry_text, emotion, score)
        return jsonify({'message': 'Entry saved', 'emotion': emotion, 'score': score})
    except Exception as e:
        logger.error(f"Submission failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/entries', methods=['GET'])
def fetch_entries():
    try:
        entries = get_all_entries()
        chart_data = {
            'labels': [entry['created_at'].strftime('%Y-%m-%d %H:%M') for entry in entries],
            'scores': [entry['score'] if entry['sentiment'] == 'happy' else -entry['score'] for entry in entries]
        }
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Fetch entries failed: {e}")
        return jsonify({'error': 'Failed to fetch entries'}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 and Render's default port (10000)
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000)), debug=True)
