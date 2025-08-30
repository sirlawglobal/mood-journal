#app.py


from flask import Flask, request, jsonify, render_template
import requests
import mysql.connector
import time
from datetime import datetime
import os
from dotenv import load_dotenv

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

app = Flask(__name__)

# MySQL configuration (update with your details)
db_config = {
    'host': 'localhost',
    'user': 'root',  # Your MySQL username
    'password': '',  # Your MySQL password
    'database': 'mood_journal'
}

HF_API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def get_db_connection():
    return mysql.connector.connect(**db_config)


# CRUD: Create - Insert new entry
def insert_entry(entry_text, sentiment, score):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "INSERT INTO entries (entry_text, sentiment, score) VALUES (%s, %s, %s)"
    cursor.execute(query, (entry_text, sentiment, score))
    conn.commit()
    cursor.close()
    conn.close()


# CRUD: Read - Get all entries for charting
def get_all_entries():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM entries ORDER BY created_at ASC"
    cursor.execute(query)
    entries = cursor.fetchall()
    cursor.close()
    conn.close()
    return entries


# Analyze sentiment using Hugging Face API (with retries + debugging)
def analyze_sentiment(text, retries=3, delay=5):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    for attempt in range(retries):
        response = requests.post(HF_API_URL, headers=headers, json=payload)

        try:
            result = response.json()
        except Exception:
            result = response.text

        print(f"[DEBUG] Hugging Face response (attempt {attempt+1}): {result}")

        if response.status_code == 200:
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                label = result[0][0]['label']
                score = result[0][0]['score']
                emotion = 'happy' if label == 'POSITIVE' else 'sad'
                return emotion, score
            else:
                raise Exception(f"Unexpected response format: {result}")

        # If model is loading, retry
        if isinstance(result, dict) and "is currently loading" in result.get("error", "").lower():
            print("[INFO] Model is loading... retrying in", delay, "seconds")
            time.sleep(delay)
            continue

        # Other errors: break immediately
        raise Exception(f"API request failed: {result}")

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
        return jsonify({'error': str(e)}), 500


@app.route('/entries', methods=['GET'])
def fetch_entries():
    entries = get_all_entries()
    chart_data = {
        'labels': [entry['created_at'].strftime('%Y-%m-%d %H:%M') for entry in entries],
        'scores': [entry['score'] if entry['sentiment'] == 'happy' else -entry['score'] for entry in entries]
    }
    return jsonify(chart_data)


if __name__ == '__main__':
    app.run(debug=True)
