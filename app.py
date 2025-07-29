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
import sqlite3
import xml.etree.ElementTree as ET

load_dotenv()
warnings.filterwarnings("ignore")

def convert_to_single_digit(value):
    """Convert a number to single digit by taking last digit of sum"""
    if pd.isna(value) or value == '' or value == '--':
        return '--'

    try:
        # Convert to string and remove any non-digit characters
        str_val = str(value).strip()
        if str_val == '--' or str_val == '':
            return '--'

        # Sum all digits
        digit_sum = sum(int(digit) for digit in str_val if digit.isdigit())

        # Return last digit of the sum
        return str(digit_sum)[-1]
    except:
        return '--'

# === ENHANCED CONFIG ===
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
ACCURACY_FILE = "prediction_accuracy.csv"
DB_FILE = "satta_analytics.db"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def handle_json_errors(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            if isinstance(result, dict):
                return jsonify(result)
            return result
        except Exception as e:
            print(f"API Error in {f.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'message': 'Internal server error'
            }), 500
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
@handle_json_errors
def get_predictions():
    """Get predictions directly from CSV file with fallback data"""
    try:
        today = datetime.now().strftime("%d/%m/%Y")
        
        # Try to read from CSV file first
        if os.path.exists(PRED_FILE):
            try:
                df_pred = pd.read_csv(PRED_FILE)
                today_preds = df_pred[df_pred['Date'] == today]

                if not today_preds.empty:
                    predictions = []
                    for _, row in today_preds.iterrows():
                        try:
                            # Convert open and close to single digits
                            open_values = [convert_to_single_digit(x.strip()) for x in str(row['Open']).split(',')]
                            close_values = [convert_to_single_digit(x.strip()) for x in str(row['Close']).split(',')]

                            predictions.append({
                                "market": row['Market'],
                                "status": "success",
                                "open": open_values,
                                "close": close_values,
                                "pattis": [x.strip() for x in str(row['Pattis']).split(',')],
                                "jodis": [x.strip() for x in str(row['Jodis']).split(',')],
                                "confidence": row.get('Confidence', 85.0) if pd.notna(row.get('Confidence')) else 85.0
                            })
                        except Exception as e:
                            print(f"Error processing prediction for {row['Market']}: {e}")
                            continue

                    return jsonify({
                        "success": True,
                        "date": today,
                        "predictions": predictions,
                        "cached": True
                    })
            except Exception as e:
                print(f"Error reading predictions file: {e}")

        # No fallback data - return empty if no predictions available
        return jsonify({
            "success": False,
            "error": "No predictions available for today",
            "predictions": [],
            "date": today
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/results')
@handle_json_errors  
def get_results():
    """Get results directly from CSV data with fallback"""
    try:
        today = datetime.now().strftime("%d/%m/%Y")
        results = []

        # Load actual results from satta_data.csv
        declared_results = {}
        if os.path.exists(DATA_FILE):
            try:
                df_results = pd.read_csv(DATA_FILE)
                df_results['Date'] = pd.to_datetime(df_results['Date'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
                today_results = df_results[df_results['Date'] == today]

                for _, row in today_results.iterrows():
                    declared_results[row['Market']] = {
                        'open': convert_to_single_digit(row['Open']),
                        'close': convert_to_single_digit(row['Close']),
                        'jodi': str(row['Jodi']).zfill(2)
                    }
            except Exception as e:
                print(f"Error reading results: {e}")

        # Load predictions for comparison
        predictions_map = {}
        try:
            if os.path.exists(PRED_FILE):
                df_pred = pd.read_csv(PRED_FILE)
                today_preds = df_pred[df_pred['Date'] == today]
                for _, row in today_preds.iterrows():
                    predictions_map[row['Market']] = {
                        'open': [convert_to_single_digit(x.strip()) for x in str(row['Open']).split(',')],
                        'close': [convert_to_single_digit(x.strip()) for x in str(row['Close']).split(',')],
                        'jodis': [x.strip() for x in str(row['Jodis']).split(',')]
                    }
        except Exception as e:
            print(f"Error loading predictions: {e}")

        # Market timings for pending status
        current_hour = datetime.now().hour
        market_timings = {
            "Time Bazar": 12,
            "Milan Day": 15,  
            "Rajdhani Day": 16,
            "Kalyan": 21,
            "Milan Night": 22,
            "Rajdhani Night": 23,
            "Main Bazar": 21
        }

        # Only use real data from CSV files
        for market in MARKETS:
            market_timing = market_timings.get(market, 21)

            if market in declared_results:
                # Result is declared from actual data
                actual = declared_results[market]
                prediction = predictions_map.get(market)

                # Check matches if prediction exists
                matches = {}
                if prediction:
                    matches = {
                        'open': actual['open'] in prediction['open'],
                        'close': actual['close'] in prediction['close'], 
                        'jodi': actual['jodi'] in prediction['jodis']
                    }

                results.append({
                    'market': market,
                    'open': actual['open'],
                    'close': actual['close'],
                    'jodi': actual['jodi'],
                    'time': f"{market_timing}:00",
                    'status': 'declared',
                    'date': today,
                    'matches': matches,
                    'has_prediction': prediction is not None
                })
            else:
                # Result is pending (no data available)
                results.append({
                    'market': market,
                    'open': '--',
                    'close': '--', 
                    'jodi': '--',
                    'time': f"{market_timing}:00",
                    'status': 'pending',
                    'date': today,
                    'matches': {},
                    'has_prediction': market in predictions_map
                })

        return jsonify({
            "success": True,
            "date": today,
            "results": results,
            "total_markets": len(results)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "results": [],
            "total_markets": 0
        })

@app.route('/sitemap.xml')
def sitemap():
    """Generate dynamic sitemap"""
    try:
        with open('sitemap.xml', 'r') as f:
            return Response(f.read(), mimetype='application/xml')
    except:
        # Fallback dynamic sitemap
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://kalyanx.replit.app/</loc>
        <lastmod>{}</lastmod>
        <changefreq>hourly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>'''.format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S+00:00'))
        return Response(xml_content, mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    """Serve robots.txt"""
    try:
        with open('robots.txt', 'r') as f:
            return Response(f.read(), mimetype='text/plain')
    except:
        return Response('User-agent: *\nAllow: /\nSitemap: https://kalyanx.replit.app/sitemap.xml', mimetype='text/plain')

@app.route('/manifest.json')
def manifest():
    """PWA manifest for mobile optimization"""
    manifest_data = {
        "name": "KalyanX - Satta Matka Analytics",
        "short_name": "KalyanX",
        "description": "Live Satta Matka results and AI predictions",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1d29",
        "theme_color": "#64b5f6",
        "icons": [
            {
                "src": "/favicon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/favicon-512.png", 
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    return jsonify(manifest_data)

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        print(f"Starting server on 0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        raise