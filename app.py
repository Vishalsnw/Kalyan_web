from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

# === CONFIG ===
warnings.filterwarnings("ignore")
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
ACCURACY_FILE = "prediction_accuracy.csv"

app = Flask(__name__)

# === UTILS ===
def patti_to_digit(patti):
    return sum(int(d) for d in str(int(patti)).zfill(3)) % 10

def generate_pattis(open_vals, close_vals):
    pattis = set()
    for val in open_vals + close_vals:
        try:
            base = int(val)
            digits = list(str(base).zfill(3))
            sorted_digits = ''.join(sorted(digits))
            pattis.add(sorted_digits)
        except:
            continue
    return sorted(pattis)[:4]

def next_prediction_date():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() == 6:  # Sunday
        return (tomorrow + timedelta(days=1)).strftime("%d/%m/%Y")
    return tomorrow.strftime("%d/%m/%Y")

# === LOAD DATA ===
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "Market", "Open", "Close", "Jodi"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Jodi"] = df["Jodi"].astype(str).str.zfill(2).str[-2:]
    return df.dropna()

# === FEATURE ENGINEERING ===
def engineer_features(df):
    df = df.sort_values("Date").copy()
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    return df.dropna(subset=["Prev_Open", "Prev_Close"])

# === MODEL TRAINING ===
def train_model(X, y):
    if len(X) < 5:
        return None
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# === TRAIN + PREDICT ===
def train_and_predict(df, market, prediction_date):
    df_market = df[df["Market"] == market].copy()
    if len(df_market) < 6:
        return None, None, None, "Insufficient data"

    df_market = engineer_features(df_market)
    if df_market.empty:
        return None, None, None, "Feature error"

    last_row = df_market.iloc[-1]
    X = df_market[["Prev_Open", "Prev_Close", "Weekday"]]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)
    y_jodi = df_market["Jodi"]

    model_open = train_model(X, y_open)
    model_close = train_model(X, y_close)
    model_jodi = train_model(X, y_jodi)

    if not all([model_open, model_close, model_jodi]):
        return None, None, None, "Model train fail"

    X_pred = pd.DataFrame([{
        "Prev_Open": last_row["Open"],
        "Prev_Close": last_row["Close"],
        "Weekday": datetime.strptime(prediction_date, "%d/%m/%Y").weekday()
    }])

    open_probs = model_open.predict_proba(X_pred)[0]
    close_probs = model_close.predict_proba(X_pred)[0]
    jodi_probs = model_jodi.predict_proba(X_pred)[0]

    open_classes = model_open.classes_
    close_classes = model_close.classes_
    jodi_classes = model_jodi.classes_

    open_vals = [open_classes[i] for i in np.argsort(open_probs)[-2:][::-1]]
    close_vals = [close_classes[i] for i in np.argsort(close_probs)[-2:][::-1]]
    jodi_vals = [jodi_classes[i] for i in np.argsort(jodi_probs)[-10:][::-1]]

    return open_vals, close_vals, jodi_vals, "Prediction successful"

# === ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    try:
        df = load_data()
        prediction_date = next_prediction_date()
        predictions = []

        for market in MARKETS:
            open_vals, close_vals, jodis, status = train_and_predict(df, market, prediction_date)

            if not open_vals or not close_vals or not jodis:
                predictions.append({
                    "market": market,
                    "status": "error",
                    "message": status
                })
                continue

            open_digits = [str(patti_to_digit(val)) for val in open_vals]
            close_digits = [str(patti_to_digit(val)) for val in close_vals]
            pattis = generate_pattis(open_vals, close_vals)

            predictions.append({
                "market": market,
                "status": "success",
                "open": open_digits,
                "close": close_digits,
                "pattis": pattis,
                "jodis": jodis,
                "date": prediction_date
            })

        return jsonify({
            "success": True,
            "date": prediction_date,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/today-results')
def get_today_results():
    try:
        today = datetime.now().strftime("%d/%m/%Y")

        # Load actual results
        try:
            df = pd.read_csv(DATA_FILE)
            df = df[df['Date'] == today]
        except:
            df = pd.DataFrame()

        # Load predictions
        try:
            pred_df = pd.read_csv(PRED_FILE)
            pred_df = pred_df[pred_df['Date'] == today]
        except:
            pred_df = pd.DataFrame()

        results = []
        for _, actual_row in df.iterrows():
            market = actual_row['Market']

            # Find corresponding prediction
            pred_row = pred_df[pred_df['Market'] == market]
            prediction = pred_row.iloc[0].to_dict() if not pred_row.empty else None

            result = {
                'market': market,
                'actual': {
                    'open': str(actual_row['Open']),
                    'close': str(actual_row['Close']),
                    'jodi': str(actual_row['Jodi'])
                },
                'prediction': prediction,
                'matches': {}
            }

            if prediction:
                # Check matches
                pred_open = [x.strip() for x in str(prediction.get('Open', '')).split(',')]
                pred_close = [x.strip() for x in str(prediction.get('Close', '')).split(',')]
                pred_jodi = [x.strip() for x in str(prediction.get('Jodis', '')).split(',')]
                pred_pattis = [x.strip() for x in str(prediction.get('Pattis', '')).split(',')]

                actual_jodi = str(actual_row['Jodi']).zfill(2)
                actual_open_digit = actual_jodi[0]
                actual_close_digit = actual_jodi[1]

                # Generate actual pattis from actual numbers
                actual_open = int(actual_row['Open'])
                actual_close = int(actual_row['Close'])
                actual_pattis = generate_pattis([actual_open], [actual_close])

                # Check patti matches
                patti_match = any(patti in pred_pattis for patti in actual_pattis)

                result['matches'] = {
                    'open': actual_open_digit in pred_open,
                    'close': actual_close_digit in pred_close,
                    'jodi': actual_jodi in pred_jodi,
                    'patti': patti_match
                }

                result['actual']['pattis'] = actual_pattis

            results.append(result)

        return jsonify({
            "success": True,
            "results": results,
            "date": today
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/generate-predictions', methods=['POST'])
def generate_predictions():
    try:
        df = load_data()
        prediction_date = next_prediction_date()

        try:
            df_existing = pd.read_csv(PRED_FILE)
        except FileNotFoundError:
            df_existing = pd.DataFrame()

        try:
            df_acc = pd.read_csv(ACCURACY_FILE)
        except FileNotFoundError:
            df_acc = pd.DataFrame()

        new_preds = []

        for market in MARKETS:
            open_vals, close_vals, jodis, status = train_and_predict(df, market, prediction_date)

            if not open_vals or not close_vals or not jodis:
                continue

            open_digits = [str(patti_to_digit(val)) for val in open_vals]
            close_digits = [str(patti_to_digit(val)) for val in close_vals]
            pattis = generate_pattis(open_vals, close_vals)

            new_preds.append({
                "Market": market,
                "Date": prediction_date,
                "Open": ", ".join(open_digits),
                "Close": ", ".join(close_digits),
                "Pattis": ", ".join(pattis),
                "Jodis": ", ".join(jodis)
            })

            df_acc = pd.concat([df_acc, pd.DataFrame([{
                "Date": prediction_date,
                "Market": market,
                "Pred_Open": open_vals,
                "Pred_Close": close_vals,
                "Pred_Jodis": jodis
            }])], ignore_index=True)

        # Save predictions
        for row in new_preds:
            df_existing = df_existing[~(
                (df_existing['Market'] == row['Market']) &
                (df_existing['Date'] == row['Date'])
            )]
        df_combined = pd.concat([df_existing, pd.DataFrame(new_preds)], ignore_index=True)
        df_combined.to_csv(PRED_FILE, index=False)
        df_acc.to_csv(ACCURACY_FILE, index=False)

        return jsonify({
            "success": True,
            "message": "Predictions generated and saved successfully",
            "count": len(new_preds)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)