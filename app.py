"""
Login Anomaly Detection Web Application
========================================
A production-ready Flask web application for detecting anomalous login attempts.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io
import base64
from datetime import datetime

from src.data_generator import LoginDataGenerator
from src.preprocessor import LoginDataPreprocessor
from src.anomaly_detector import LoginAnomalyDetector
from src.visualizer import AnomalyVisualizer
from src.config import MODELS_DIR, DATA_DIR, OUTPUTS_DIR
from src.logger import logger
from database import init_db, insert_log

app = Flask(__name__)
# Ensure MAX_CONTENT_LENGTH is set for file uploads
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for loaded models
detector = None
preprocessor = None


def get_risk_level(score):
    """Determine risk level based on anomaly score."""
    if score < -0.1:
        return "Critical"
    elif score < -0.05:
        return "High"
    elif score < 0:
        return "Medium"
    else:
        return "Low"


def load_models():
    """Load trained models on application startup."""
    global detector, preprocessor

    try:
        model_path = MODELS_DIR / "isolation_forest_model.pkl"
        scaler_path = MODELS_DIR / "scaler.pkl"

        if model_path.exists() and scaler_path.exists():
            detector = LoginAnomalyDetector()
            detector.load_model(model_path)

            preprocessor = LoginDataPreprocessor()
            preprocessor.load_scaler(scaler_path)

            logger.info("Models loaded successfully")
            return True
        else:
            logger.warning("Models not found. Please train the model first.")
            return False
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False


# Initialize database (safe call)
init_db()


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def train_model():
    """Train a new model with synthetic data."""
    try:
        data = request.get_json()
        n_samples = int(data.get("n_samples", 1000))
        anomaly_ratio = float(data.get("anomaly_ratio", 0.05))

        if n_samples < 100 or n_samples > 100000:
            return (
                jsonify({"error": "Sample size must be between 100 and 100,000"}),
                400,
            )

        if anomaly_ratio < 0.01 or anomaly_ratio > 0.5:
            return jsonify({"error": "Anomaly ratio must be between 0.01 and 0.5"}), 400

        logger.info(
            f"Generating {n_samples} samples with {anomaly_ratio*100}% anomalies"
        )
        generator = LoginDataGenerator(n_samples=n_samples, anomaly_ratio=anomaly_ratio)
        df = generator.generate_dataset()
        generator.save_dataset(df)

        global preprocessor
        preprocessor = LoginDataPreprocessor()
        X_scaled, y = preprocessor.preprocess(df, fit=True)
        preprocessor.save_scaler()

        global detector
        detector = LoginAnomalyDetector(contamination=anomaly_ratio)
        detector.train(X_scaled)
        detector.save_model()

        predictions, scores = detector.predict_with_scores(X_scaled)
        metrics = detector.evaluate(X_scaled, y)

        return jsonify(
            {
                "success": True,
                "message": "Model trained successfully",
                "metrics": {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1_score"]),
                },
                "samples_generated": n_samples,
                "anomalies_detected": int((predictions == -1).sum()),
            }
        )

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict anomalies for uploaded data or single record."""
    try:
        if not detector or not preprocessor:
            return (
                jsonify({"error": "Model not trained. Please train the model first."}),
                400,
            )

        # Check if file upload or JSON data
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            # Read CSV file
            df = pd.read_csv(file)
            is_single = False

        else:
            # Single record from JSON
            data = request.get_json()
            df = pd.DataFrame(
                [
                    {
                        "login_hour": int(data.get("login_hour", 12)),
                        "login_attempts": int(data.get("login_attempts", 1)),
                        "ip_frequency": int(data.get("ip_frequency", 10)),
                        "device_type": int(data.get("device_type", 1)),
                        "login_success": int(data.get("login_success", 1)),
                    }
                ]
            )
            is_single = True

        # Preprocess
        X_scaled, _ = preprocessor.preprocess(df, fit=False)

        # Predict
        predictions, scores = detector.predict_with_scores(X_scaled)

        # Prepare results
        results = []
        for i in range(len(predictions)):
            prediction_label = "Anomaly" if predictions[i] == -1 else "Normal"
            risk = get_risk_level(scores[i])
            score_val = float(scores[i])
            row_data = df.iloc[i].to_dict()

            results.append(
                {
                    "index": i,
                    "prediction": prediction_label,
                    "anomaly_score": score_val,
                    "risk_level": risk,
                    "features": row_data,
                }
            )

            # Log single predictions to database
            if is_single:
                try:
                    # insert_log(input_data, risk_score, risk_level)
                    insert_log(row_data, score_val, risk)
                except Exception as log_err:
                    logger.warning(f"Failed to log single prediction: {log_err}")

        return jsonify(
            {
                "success": True,
                "total_records": len(predictions),
                "anomalies_detected": int((predictions == -1).sum()),
                "normal_logins": int((predictions == 1).sum()),
                "results": results,
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualize", methods=["POST"])
def visualize():
    """Generate visualization for current data."""
    try:
        if not detector or not preprocessor:
            return (
                jsonify({"error": "Model not trained. Please train the model first."}),
                400,
            )

        # Load latest data
        data_path = DATA_DIR / "login_data.csv"
        if not data_path.exists():
            return (
                jsonify({"error": "No data available. Please train the model first."}),
                400,
            )

        df = pd.read_csv(data_path)
        X_scaled, y = preprocessor.preprocess(df, fit=False)
        predictions, scores = detector.predict_with_scores(X_scaled)

        # Generate visualizations
        visualizer = AnomalyVisualizer()

        # Create dashboard
        dashboard_path = OUTPUTS_DIR / "dashboard.png"
        visualizer.create_dashboard(
            df[preprocessor.feature_columns], predictions, scores, y, dashboard_path
        )

        # Read image and convert to base64
        with open(dashboard_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"success": True, "image": f"data:image/png;base64,{img_data}"})

    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get current model statistics."""
    try:
        if not detector or not preprocessor:
            return jsonify({"error": "Model not trained"}), 400

        data_path = DATA_DIR / "login_data.csv"
        if not data_path.exists():
            return jsonify({"error": "No data available"}), 400

        df = pd.read_csv(data_path)
        X_scaled, y = preprocessor.preprocess(df, fit=False)
        predictions, scores = detector.predict_with_scores(X_scaled)

        return jsonify(
            {
                "success": True,
                "total_records": len(df),
                "normal_count": int((predictions == 1).sum()),
                "anomaly_count": int((predictions == -1).sum()),
                "anomaly_percentage": float(
                    (predictions == -1).sum() / len(predictions) * 100
                ),
                "model_contamination": float(detector.contamination),
                "model_estimators": int(detector.n_estimators),
            }
        )

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


# Load models on import so Gunicorn workers have them ready
load_models()

if __name__ == "__main__":
    # Use PORT environment variable if available (default 10000)
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 for external access
    app.run(host="0.0.0.0", port=port, debug=False)
