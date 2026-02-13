"""
Anomaly Detector Module
========================
Implements Isolation Forest for login anomaly detection.

Author: Senior Data Scientist & Cybersecurity Engineer
Purpose: Train and deploy anomaly detection model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from .config import MODELS_DIR, CONTAMINATION, N_ESTIMATORS, RANDOM_STATE
from .logger import logger


class LoginAnomalyDetector:
    """
    Anomaly detection system using Isolation Forest algorithm.

    Key Concepts:
    -------------
    Isolation Forest works by:
    1. Randomly selecting a feature
    2. Randomly selecting a split value between min and max
    3. Recursively partitioning data
    4. Anomalies are isolated in fewer splits (shorter path length)

    Anomaly Score:
    - Based on average path length across all trees
    - Shorter paths → Higher anomaly score → Anomaly (-1)
    - Longer paths → Lower anomaly score → Normal (1)
    """

    def __init__(
        self,
        contamination: float = CONTAMINATION,
        n_estimators: int = N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Initialize Isolation Forest model.

        Parameters:
        -----------
        contamination : float (default=0.05)
            Expected proportion of outliers in dataset.
            - 0.05 = 5% of data expected to be anomalies
            - Range: 0.0 to 0.5
            - Too low: Misses anomalies
            - Too high: Too many false positives
            - Security sweet spot: 0.01 to 0.1

        n_estimators : int (default=100)
            Number of isolation trees in the forest.
            - More trees = Better performance but slower
            - 100 is good balance for most cases
            - Production: Consider 200-300 for better accuracy

        random_state : int (default=42)
            Random seed for reproducibility
        """
        self.contamination: float = contamination
        self.n_estimators: int = n_estimators
        self.random_state: int = random_state

        # Initialize model
        self.model: IsolationForest = IsolationForest(
            contamination=contamination,  # Float value between 0.0 and 0.5
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples="auto",  # Use all samples for training
            max_features=1.0,  # Use all features
            bootstrap=False,  # Don't bootstrap samples
            n_jobs=-1,  # Use all CPU cores
            verbose=0,
        )

        self.is_trained: bool = False

    def train(self, X):
        """
        Train Isolation Forest model.

        Training Process:
        1. Build multiple isolation trees
        2. Each tree randomly partitions data
        3. Calculate average path length for each sample
        4. Determine anomaly threshold based on contamination

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training features (scaled)

        Returns:
        --------
        self : Returns self for method chaining
        """
        logger.info("Training Isolation Forest...")
        logger.info("Configuration:")
        logger.info(f"    - Number of trees: {self.n_estimators}")
        logger.info(
            f"    - Contamination: {self.contamination} ({self.contamination*100}%)"
        )
        logger.info(f"    - Training samples: {X.shape[0]}")
        logger.info(f"    - Features: {X.shape[1]}")

        # Train model
        self.model.fit(X)
        self.is_trained = True

        logger.info("Model trained successfully")

        return self

    def predict(self, X):
        """
        Predict anomalies in new data.

        Output:
        -------
        1  = Normal login
        -1 = Anomalous login (SECURITY ALERT!)

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to predict (scaled)

        Returns:
        --------
        np.ndarray : Predictions (1 or -1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_with_scores(self, X):
        """
        Predict anomalies with anomaly scores.

        Anomaly Score (decision_function):
        - Negative values → Anomalies
        - Positive values → Normal
        - More negative → Higher confidence anomaly
        - More positive → Higher confidence normal

        Use cases:
        - Prioritize alerts by severity
        - Set custom thresholds
        - Create risk levels (low/medium/high)

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Features to predict (scaled)

        Returns:
        --------
        tuple : (predictions, scores)
            - predictions: Binary labels (1 or -1)
            - scores: Anomaly scores (continuous)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)

        return predictions, scores

    def evaluate(self, X, y_true):
        """
        Evaluate model performance against ground truth.

        Metrics:
        --------
        - Accuracy: Overall correctness
        - Precision: Of predicted anomalies, how many are real?
        - Recall: Of real anomalies, how many did we catch?
        - F1-Score: Harmonic mean of precision and recall

        Security Context:
        - High Recall: Don't miss attacks (critical!)
        - High Precision: Reduce alert fatigue
        - Trade-off: Adjust contamination parameter

        Parameters:
        -----------
        X : np.ndarray
            Features (scaled)
        y_true : pd.Series or np.ndarray
            Ground truth labels ('normal' or 'anomaly')

        Returns:
        --------
        dict : Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Predict
        y_pred = self.predict(X)

        # Convert labels to match prediction format
        # 'normal' → 1, 'anomaly' → -1
        y_true_binary = np.where(y_true == "normal", 1, -1)

        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred)

        logger.info("=== Model Evaluation ===")
        logger.info(f"Accuracy: {accuracy:.4f}")

        logger.info("--- Confusion Matrix ---")
        cm = confusion_matrix(y_true_binary, y_pred, labels=[1, -1])
        # logger.info(f"\n{cm}") # Simple log

        logger.info("--- Classification Report ---")
        logger.info(
            "\n"
            + classification_report(
                y_true_binary,
                y_pred,
                target_names=["Normal", "Anomaly"],
                labels=[1, -1],
            )
        )

        # Calculate additional metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        metrics = {
            "accuracy": accuracy,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }

        return metrics

    def get_anomaly_summary(self, X, predictions, scores, top_n=10):
        """
        Get summary of detected anomalies.

        Parameters:
        -----------
        X : pd.DataFrame
            Original features (before scaling)
        predictions : np.ndarray
            Model predictions
        scores : np.ndarray
            Anomaly scores
        top_n : int
            Number of top anomalies to return

        Returns:
        --------
        pd.DataFrame : Top anomalies sorted by severity
        """
        # Create results dataframe
        results = X.copy()
        results["prediction"] = predictions
        results["anomaly_score"] = scores
        results["prediction_label"] = np.where(predictions == 1, "NORMAL", "ANOMALY")

        # Filter anomalies
        anomalies = results[results["prediction"] == -1].copy()

        # Sort by severity (most negative score = most anomalous)
        anomalies = anomalies.sort_values("anomaly_score").head(top_n)

        return anomalies

    def save_model(self, filepath=None):
        """
        Save trained model to disk.

        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if filepath is None:
            filepath = MODELS_DIR / "isolation_forest_model.pkl"

        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath=None):
        """
        Load pre-trained model from disk.

        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        if filepath is None:
            filepath = MODELS_DIR / "isolation_forest_model.pkl"

        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # This is a standalone test
    # In production, use main.py for complete pipeline

    print("=== Anomaly Detector Module Test ===")
    print("For full pipeline, run: python main.py")

    # Generate sample data for testing
    from .data_generator import LoginDataGenerator
    from .preprocessor import LoginDataPreprocessor

    # Generate data
    generator = LoginDataGenerator(n_samples=1000, anomaly_ratio=0.05)
    df = generator.generate_dataset()

    # Preprocess
    preprocessor = LoginDataPreprocessor()
    X_scaled, y = preprocessor.preprocess(df, fit=True)

    # Train model
    detector = LoginAnomalyDetector(contamination=0.05)
    detector.train(X_scaled)

    # Evaluate
    metrics = detector.evaluate(X_scaled, y)

    # Predict with scores
    predictions, scores = detector.predict_with_scores(X_scaled)

    # Get top anomalies
    print("\n=== Top 10 Anomalies ===")
    anomalies = detector.get_anomaly_summary(
        df[preprocessor.feature_columns], predictions, scores, top_n=10
    )
    print(anomalies)
