"""
Data Preprocessor Module
=========================
Handles data preprocessing and feature engineering for anomaly detection.

Author: Senior Data Scientist & Cybersecurity Engineer
Purpose: Prepare login data for machine learning model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional, Tuple, Union, List
from .config import FEATURE_COLUMNS, MODELS_DIR
from .logger import logger


class LoginDataPreprocessor:
    """
    Preprocesses login data for anomaly detection.

    Steps:
    1. Feature selection
    2. Encoding categorical variables
    3. Feature scaling (StandardScaler)
    4. Handling missing values
    """

    def __init__(self) -> None:
        """Initialize preprocessor with scaler."""
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = [
            "login_hour",
            "login_attempts",
            "ip_frequency",
            "device_type",
            "login_success",
        ]
        self.is_fitted: bool = False

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame : DataFrame with selected features
        """
        # Keep only modeling features
        if "label" in df.columns:
            # Training data - keep label for evaluation
            return df[self.feature_columns + ["label"]].copy()
        else:
            # Prediction data - no label
            return df[self.feature_columns].copy()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy:
        - login_hour: Fill with median (most common hour)
        - login_attempts: Fill with 1 (default)
        - ip_frequency: Fill with median
        - device_type: Fill with mode (most common device)
        - login_success: Fill with 0 (failed - conservative approach)

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame : DataFrame with imputed values
        """
        df_imputed = df.copy()

        if df_imputed[self.feature_columns].isnull().sum().sum() > 0:
            logger.warning("Missing values detected. Applying imputation...")

            # Imputation strategies
            if "login_hour" in df_imputed.columns:
                df_imputed["login_hour"] = df_imputed["login_hour"].fillna(
                    df_imputed["login_hour"].median()
                )
            if "login_attempts" in df_imputed.columns:
                df_imputed["login_attempts"] = df_imputed["login_attempts"].fillna(1)
            if "ip_frequency" in df_imputed.columns:
                df_imputed["ip_frequency"] = df_imputed["ip_frequency"].fillna(
                    df_imputed["ip_frequency"].median()
                )
            if "device_type" in df_imputed.columns:
                # mode() returns a Series, take the first element
                mode_val = df_imputed["device_type"].mode()
                fill_val = mode_val[0] if not mode_val.empty else 0
                df_imputed["device_type"] = df_imputed["device_type"].fillna(fill_val)
            if "login_success" in df_imputed.columns:
                df_imputed["login_success"] = df_imputed["login_success"].fillna(0)

            logger.info("Missing values imputed")

        return df_imputed

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.

        Note: In this dataset, device_type is already binary (0/1).
        This method is a placeholder for future categorical features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame : DataFrame with encoded features
        """
        # device_type is already encoded (0=Mobile, 1=Desktop)
        # If we had other categorical features, we'd use one-hot encoding here

        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.

        Why StandardScaler?
        - Isolation Forest is distance-based
        - Features have different scales (hour: 0-23, attempts: 1-20+)
        - Standardization ensures equal feature importance

        Formula: z = (x - μ) / σ
        - μ = mean
        - σ = standard deviation

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            If True, fit scaler on data (training)
            If False, use existing scaler (prediction)

        Returns:
        --------
        np.ndarray : Scaled feature array
        """
        features = df[self.feature_columns].values

        if fit:
            logger.info("Fitting scaler on training data...")
            scaled_features = self.scaler.fit_transform(features)
            self.is_fitted = True
            logger.info("Scaler fitted and features scaled")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_features = self.scaler.transform(features)
            logger.info("Features scaled using existing scaler")

        return scaled_features

    def preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """
        Complete preprocessing pipeline.

        Pipeline:
        1. Select features
        2. Handle missing values
        3. Encode categorical features
        4. Scale features

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            If True, fit scaler (training mode)
            If False, use existing scaler (prediction mode)

        Returns:
        --------
        tuple : (scaled_features, labels)
            - scaled_features: np.ndarray of scaled features
            - labels: pd.Series of ground truth labels (if available)
        """
        logger.info("Starting preprocessing pipeline")

        # Step 1: Select features
        df_features = self.select_features(df)

        # Step 2: Handle missing values
        df_features = self.handle_missing_values(df_features)

        # Step 3: Encode features
        df_features = self.encode_features(df_features)

        # Step 4: Scale features
        scaled_features = self.scale_features(df_features, fit=fit)

        # Extract labels if available
        labels = df["label"] if "label" in df.columns else None

        logger.info("Preprocessing complete")
        logger.info(f"    - Features shape: {scaled_features.shape}")

        return scaled_features, labels

    def save_scaler(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Save fitted scaler for future use.

        Parameters:
        -----------
        filepath : str or Path, optional
            Path to save scaler
        """
        if filepath is None:
            filepath = MODELS_DIR / "scaler.pkl"

        filepath = Path(filepath)

        if not self.is_fitted:
            # Try to verify if it's really not fitted or just a new instance
            # checks if mean_ attribute exists which is set during fit
            if not hasattr(self.scaler, "mean_"):
                raise ValueError("Scaler not fitted. Cannot save.")

        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Load pre-fitted scaler.

        Parameters:
        -----------
        filepath : str or Path, optional
            Path to load scaler from
        """
        if filepath is None:
            filepath = MODELS_DIR / "scaler.pkl"

        filepath = Path(filepath)

        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame : Statistical summary
        """
        return df[self.feature_columns].describe()


# Example usage
if __name__ == "__main__":
    # Load sample data
    try:
        from .config import DATA_DIR

        data_path = DATA_DIR / "login_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)

            # Initialize preprocessor
            preprocessor = LoginDataPreprocessor()

            # Preprocess data
            X_scaled, y = preprocessor.preprocess(df, fit=True)

            # Display results
            print("\n=== Scaled Features (first 5 rows) ===")
            print(X_scaled[:5])

            print("\n=== Feature Statistics ===")
            print(preprocessor.get_feature_stats(df))

            # Save scaler
            preprocessor.save_scaler()
        else:
            print("Data file not found. Run data_generator.py first.")
    except ImportError:
        # Standalone run might fail structure imports
        pass
