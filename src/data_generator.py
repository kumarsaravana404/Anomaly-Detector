"""
Data Generator Module
=====================
Generates realistic synthetic login data for anomaly detection training.

Author: Senior Data Scientist & Cybersecurity Engineer
Purpose: Create training data with normal and anomalous login patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from .config import (
    DATA_DIR,
    OUTPUTS_DIR,
    N_SAMPLES,
    ANOMALY_RATIO,
    RANDOM_STATE,
    FEATURE_COLUMNS,
)
from .logger import logger


class LoginDataGenerator:
    """
    Generates synthetic login data with realistic patterns.

    Normal patterns:
    - Business hours (9 AM - 5 PM)
    - 1-2 login attempts
    - Frequent IP addresses
    - Consistent device usage
    - High success rate

    Anomalous patterns:
    - Odd hours (late night/early morning)
    - Multiple failed attempts (brute force)
    - Rare/new IP addresses
    - Device switching
    - Multiple failures
    """

    def __init__(self, n_samples=1000, anomaly_ratio=0.05, random_state=42):
        """
        Initialize the data generator.

        Parameters:
        -----------
        n_samples : int
            Total number of login records to generate
        anomaly_ratio : float
            Proportion of anomalous logins (0.0 to 1.0)
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.anomaly_ratio = anomaly_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        # Define feature columns
        self.feature_columns = [
            "login_hour",
            "login_attempts",
            "ip_frequency",
            "device_type",
            "login_success",
        ]

        # Calculate split
        self.n_anomalies = int(n_samples * anomaly_ratio)
        self.n_normal = n_samples - self.n_anomalies

    def generate_normal_logins(self):
        """
        Generate normal login patterns.

        Returns:
        --------
        pd.DataFrame : Normal login records
        """
        data = {
            "login_hour": np.random.choice(
                range(9, 18),  # Business hours: 9 AM - 5 PM
                size=self.n_normal,
                p=[
                    0.08,
                    0.12,
                    0.15,
                    0.18,
                    0.15,
                    0.12,
                    0.10,
                    0.06,
                    0.04,
                ],  # Peak at midday
            ),
            "login_attempts": np.random.choice(
                [1, 2, 3],  # Mostly 1-2 attempts
                size=self.n_normal,
                p=[0.7, 0.25, 0.05],  # 70% succeed on first try
            ),
            "ip_frequency": np.random.choice(
                range(10, 100), size=self.n_normal  # Frequent IPs (seen 10-100 times)
            ),
            "device_type": np.random.choice(
                [0, 1],  # 0=Mobile, 1=Desktop
                size=self.n_normal,
                p=[0.4, 0.6],  # 60% desktop in business environment
            ),
            "login_success": np.random.choice(
                [0, 1],  # 0=Failed, 1=Success
                size=self.n_normal,
                p=[0.05, 0.95],  # 95% success rate for normal users
            ),
        }

        df = pd.DataFrame(data)
        df["label"] = "normal"  # Ground truth for evaluation
        return df

    def generate_anomalous_logins(self):
        """
        Generate anomalous login patterns representing various attack scenarios.

        Attack types simulated:
        1. Brute force: Multiple failed attempts
        2. Credential stuffing: Odd hours + new IPs
        3. Account takeover: Device switching + unusual hours
        4. Insider threat: Unusual access patterns

        Returns:
        --------
        pd.DataFrame : Anomalous login records
        """
        anomalies = []

        # Type 1: Brute Force Attack (40% of anomalies)
        n_brute_force = int(self.n_anomalies * 0.4)
        brute_force = {
            "login_hour": np.random.choice(range(0, 24), size=n_brute_force),
            "login_attempts": np.random.choice(
                range(5, 20), size=n_brute_force  # 5-20 attempts (suspicious)
            ),
            "ip_frequency": np.random.choice(
                range(1, 5), size=n_brute_force  # New/rare IPs
            ),
            "device_type": np.random.choice([0, 1], size=n_brute_force),
            "login_success": np.zeros(n_brute_force, dtype=int),  # All failed
        }
        anomalies.append(pd.DataFrame(brute_force))

        # Type 2: Odd Hours Access (30% of anomalies)
        n_odd_hours = int(self.n_anomalies * 0.3)
        odd_hours = {
            "login_hour": np.random.choice(
                [0, 1, 2, 3, 4, 5, 22, 23], size=n_odd_hours  # Late night/early morning
            ),
            "login_attempts": np.random.choice(range(1, 4), size=n_odd_hours),
            "ip_frequency": np.random.choice(
                range(1, 10), size=n_odd_hours  # Unfamiliar IPs
            ),
            "device_type": np.random.choice([0, 1], size=n_odd_hours),
            "login_success": np.random.choice(
                [0, 1], size=n_odd_hours, p=[0.3, 0.7]  # Some succeed (more dangerous)
            ),
        }
        anomalies.append(pd.DataFrame(odd_hours))

        # Type 3: Rapid Device Switching (20% of anomalies)
        n_device_switch = int(self.n_anomalies * 0.2)
        device_switch = {
            "login_hour": np.random.choice(range(0, 24), size=n_device_switch),
            "login_attempts": np.random.choice(range(2, 6), size=n_device_switch),
            "ip_frequency": np.random.choice(range(1, 15), size=n_device_switch),
            "device_type": np.random.choice([0, 1], size=n_device_switch),
            "login_success": np.random.choice(
                [0, 1], size=n_device_switch, p=[0.4, 0.6]
            ),
        }
        anomalies.append(pd.DataFrame(device_switch))

        # Type 4: High-frequency failed attempts (10% of anomalies)
        n_high_freq = self.n_anomalies - n_brute_force - n_odd_hours - n_device_switch
        high_freq = {
            "login_hour": np.random.choice(range(0, 24), size=n_high_freq),
            "login_attempts": np.random.choice(
                range(8, 25), size=n_high_freq  # Very high attempts
            ),
            "ip_frequency": np.random.choice(range(1, 3), size=n_high_freq),
            "device_type": np.random.choice([0, 1], size=n_high_freq),
            "login_success": np.zeros(n_high_freq, dtype=int),
        }
        anomalies.append(pd.DataFrame(high_freq))

        # Combine all anomaly types
        df_anomalies = pd.concat(anomalies, ignore_index=True)
        df_anomalies["label"] = "anomaly"  # Ground truth

        return df_anomalies

    def generate_dataset(self):
        """
        Generate complete dataset with normal and anomalous logins.

        Returns:
        --------
        pd.DataFrame : Complete dataset with shuffled records
        """
        logger.info(f"Generating {self.n_normal} normal logins...")
        normal_data = self.generate_normal_logins()

        logger.info(f"Generating {self.n_anomalies} anomalous logins...")
        anomaly_data = self.generate_anomalous_logins()

        # Combine and shuffle
        df = pd.concat([normal_data, anomaly_data], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Add timestamp (for realism, not used in model)
        base_date = datetime.now() - timedelta(days=30)
        df["timestamp"] = [
            base_date
            + timedelta(
                days=random.randint(0, 30),
                hours=row["login_hour"],
                minutes=random.randint(0, 59),
            )
            for _, row in df.iterrows()
        ]

        # Add user_id (for realism)
        df["user_id"] = np.random.choice(range(1000, 2000), size=len(df))

        # Reorder columns
        df = df[
            [
                "timestamp",
                "user_id",
                "login_hour",
                "login_attempts",
                "ip_frequency",
                "device_type",
                "login_success",
                "label",
            ]
        ]

        logger.info(f"Generated {len(df)} total login records")
        logger.info(
            f"    - Normal: {len(df[df['label'] == 'normal'])} ({len(df[df['label'] == 'normal'])/len(df)*100:.1f}%)"
        )
        logger.info(
            f"    - Anomalies: {len(df[df['label'] == 'anomaly'])} ({len(df[df['label'] == 'anomaly'])/len(df)*100:.1f}%)"
        )

        return df

    def save_dataset(self, df, filepath=None):
        """
        Save dataset to CSV file.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to save
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = DATA_DIR / "login_data.csv"

        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Generate data
    generator = LoginDataGenerator(n_samples=1000, anomaly_ratio=0.05)
    df = generator.generate_dataset()

    # Display sample
    print("\n=== Sample Data ===")
    print(df.head(10))

    # Display statistics
    print("\n=== Feature Statistics ===")
    print(df.describe())

    # Save
    generator.save_dataset(df)
