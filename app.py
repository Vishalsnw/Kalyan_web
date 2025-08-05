import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

def convert_to_single_digit(value):
    if pd.isna(value) or value == '' or value == '--':
        return '--'
    try:
        digits = [int(d) for d in str(value) if d.isdigit()]
        return str(sum(digits))[-1] if digits else '--'
    except:
        return '--'

MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret')

def handle_json_errors(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            return jsonify(result) if isinstance(result, dict) else result
        except Exception as e:
            print(f"[ERROR] {f.__name__}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    return wrapper

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
@handle_json_errors
def get_predictions():
    today = datetime.now().strftime("%d/%m/%Y")
    if not os.path.exists(PRED_FILE):
        print("Prediction file not found.")
        return {"success": False, "error": "Prediction file missing", "predictions": [], "date": today}

    df = pd.read_csv(PRED_FILE, encoding='utf-8')
    df = df[df['Date'] == today]

    if df.empty:
        print("No predictions found for today.")
        return {"success": False, "error": "No predictions available for today", "predictions": [], "date": today}

    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "market": row['Market'],
            "status": "success",
            "open": [convert_to_single_digit(x.strip()) for x in str(row['Open']).split(',')],
            "close": [convert_to_single_digit(x.strip()) for x in str(row['Close']).split(',')],
            "pattis": [x.strip() for x in str(row['Pattis']).split(',')],
            "jodis": [x.strip() for x in str(row['Jodis']).split(',')],
            "confidence": row.get('Confidence', 85.0)
        })

    return {"success": True, "date": today, "predictions": predictions}

@app.route('/api/results')
@handle_json_errors
def get_results():
    today = datetime.now().strftime("%d/%m/%Y")
    results = []
    declared = {}

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, encoding='utf-8')
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
        df_today = df[df['Date'] == today]

        for _, row in df_today.iterrows():
            jodi = str(row['Jodi']).zfill(2) if str(row['Jodi']).isdigit() else '--'
            declared[row['Market']] = {
                'open': convert_to_single_digit(row['Open']),
                'close': convert_to_single_digit(row['Close']),
                'jodi': jodi
            }

    pred_map = {}
    if os.path.exists(PRED_FILE):
        dfp = pd.read_csv(PRED_FILE)
        dfp = dfp[dfp['Date'] == today]
        for _, row in dfp.iterrows():
            pred_map[row['Market']] = {
                'open': [convert_to_single_digit(x.strip()) for x in str(row['Open']).split(',')],
                'close': [convert_to_single_digit(x.strip()) for x in str(row['Close']).split(',')],
                'jodis': [x.strip() for x in str(row['Jodis']).split(',')]
            }

    timings = {
        "Time Bazar": 12, "Milan Day": 15, "Rajdhani Day": 16,
        "Kalyan": 21, "Milan Night": 22, "Rajdhani Night": 23, "Main Bazar": 21
    }

    for market in MARKETS:
        timing = timings.get(market, 21)
        if market in declared:
            actual = declared[market]
            pred = pred_map.get(market)
            match = {
                'open': actual['open'] in pred['open'] if pred else False,
                'close': actual['close'] in pred['close'] if pred else False,
                'jodi': actual['jodi'] in pred['jodis'] if pred else False
            } if pred else {}
            results.append({
                'market': market,
                'open': actual['open'],
                'close': actual['close'],
                'jodi': actual['jodi'],
                'time': f"{timing}:00",
                'status': 'declared',
                'date': today,
                'matches': match,
                'has_prediction': pred is not None
            })
        else:
            results.append({
                'market': market,
                'open': '--', 'close': '--', 'jodi': '--',
                'time': f"{timing}:00",
                'status': 'pending',
                'date': today,
                'matches': {},
                'has_prediction': market in pred_map
            })

    return {
        "success": True,
        "date": today,
        "results": results,
        "total_markets": len(results)
    }

@app.route('/sitemap.xml')
def sitemap():
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://kalyanx.replit.app/</loc>
        <lastmod>{datetime.now().strftime('%Y-%m-%dT%H:%M:%S+00:00')}</lastmod>
        <changefreq>hourly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>'''
    return Response(xml, mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    return Response('User-agent: *\nAllow: /\nSitemap: https://kalyanx.replit.app/sitemap.xml', mimetype='text/plain')

@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "KalyanX - Satta Matka Analytics",
        "short_name": "KalyanX",
        "description": "Live Satta Matka results and AI predictions",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1d29",
        "theme_color": "#64b5f6",
        "icons": [
            {"src": "/favicon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/favicon-512.png", "sizes": "512x512", "type": "image/png"}
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
