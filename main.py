import sys
import numpy as np
from pathlib import Path
from src.config import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    N_SAMPLES,
    ANOMALY_RATIO,
    RANDOM_STATE,
    CONTAMINATION,
    N_ESTIMATORS,
    FEATURE_COLUMNS,
)
from src.logger import logger

# Import modules
try:
    from src.data_generator import LoginDataGenerator
    from src.preprocessor import LoginDataPreprocessor
    from src.anomaly_detector import LoginAnomalyDetector
    from src.visualizer import AnomalyVisualizer
except ImportError:
    # Fall back to relative import (when imported as module)
    from .src.data_generator import LoginDataGenerator
    from .src.preprocessor import LoginDataPreprocessor
    from .src.anomaly_detector import LoginAnomalyDetector
    from .src.visualizer import AnomalyVisualizer


def create_directories() -> None:
    """Create necessary project directories."""
    directories = [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Project directories created/verified")


def main():
    """Main execution pipeline."""
    logger.info("Starting Login Anomaly Detection System")

    # Create directories
    create_directories()

    # ========================================
    # STEP 1: Generate Synthetic Data
    # ========================================
    logger.info("STEP 1: Data Generation")

    generator = LoginDataGenerator(
        n_samples=N_SAMPLES,
        anomaly_ratio=ANOMALY_RATIO,
        random_state=RANDOM_STATE,
    )

    df = generator.generate_dataset()
    generator.save_dataset(df)

    logger.info(f"Dataset Summary: {len(df)} records generated")

    # ========================================
    # STEP 2: Data Preprocessing
    # ========================================
    logger.info("STEP 2: Data Preprocessing")

    preprocessor = LoginDataPreprocessor()
    X_scaled, y_true = preprocessor.preprocess(df, fit=True)

    # Save scaler
    preprocessor.save_scaler()

    # ========================================
    # STEP 3: Train Anomaly Detection Model
    # ========================================
    logger.info("STEP 3: Model Training")

    detector = LoginAnomalyDetector(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    detector.train(X_scaled)
    detector.save_model()

    # ========================================
    # STEP 4: Detect Anomalies
    # ========================================
    logger.info("STEP 4: Anomaly Detection")

    predictions, scores = detector.predict_with_scores(X_scaled)

    # Count detections
    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()

    logger.info(f"Detection Results: {n_normal} Normal, {n_anomalies} Anomalies")
    logger.info(f"Detection Rate: {n_anomalies/len(predictions)*100:.2f}%")

    # ========================================
    # STEP 5: Model Evaluation
    # ========================================
    logger.info("STEP 5: Model Evaluation")

    metrics = detector.evaluate(X_scaled, y_true)

    # ========================================
    # STEP 6: Analyze Top Anomalies
    # ========================================
    logger.info("STEP 6: Top Anomalies Analysis")

    top_anomalies = detector.get_anomaly_summary(
        df[preprocessor.feature_columns], predictions, scores, top_n=10
    )

    logger.info(
        f"Top 10 Most Suspicious Logins:\n{top_anomalies.to_string(index=False)}"
    )

    # Save anomalies to CSV
    all_results = df.copy()
    all_results["prediction"] = predictions
    all_results["anomaly_score"] = scores
    all_results["prediction_label"] = np.where(predictions == 1, "NORMAL", "ANOMALY")

    results_path = OUTPUTS_DIR / "detection_results.csv"
    all_results.to_csv(results_path, index=False)
    logger.info(f"Full results saved to {results_path}")

    # ========================================
    # STEP 7: Visualization
    # ========================================
    logger.info("STEP 7: Visualization")

    visualizer = AnomalyVisualizer()

    visualizer.plot_anomalies_2d(
        df[preprocessor.feature_columns],
        predictions,
        scores,
        feature_x="login_hour",
        feature_y="login_attempts",
    )

    visualizer.plot_feature_distributions(
        df[preprocessor.feature_columns],
        predictions,
    )

    visualizer.plot_anomaly_scores(scores, predictions)

    visualizer.plot_confusion_matrix(y_true, predictions)

    visualizer.create_dashboard(
        df[preprocessor.feature_columns],
        predictions,
        scores,
        y_true=y_true,
    )

    logger.info("PROJECT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)
