"""
Visualizer Module
=================
Creates visualizations for anomaly detection results.

Author: Senior Data Scientist & Cybersecurity Engineer
Purpose: Visualize normal vs. anomalous login patterns
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from .config import OUTPUTS_DIR
from .logger import logger


class AnomalyVisualizer:
    """
    Visualization utilities for login anomaly detection.

    Provides:
    - Scatter plots showing anomalies
    - Feature distribution comparisons
    - Anomaly score distributions
    - Confusion matrix heatmaps
    """

    def __init__(self, style="darkgrid"):
        """
        Initialize visualizer with style settings.

        Parameters:
        -----------
        style : str
            Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        """
        sns.set_style(style)
        self.colors = {
            "normal": "#2E86AB",  # Blue
            "anomaly": "#A23B72",  # Red/Pink
            "background": "#F8F9FA",  # Light gray
        }

    def plot_anomalies_2d(
        self,
        X,
        predictions,
        scores,
        feature_x="login_hour",
        feature_y="login_attempts",
        filepath=None,
    ):
        """
        Create 2D scatter plot showing anomalies.

        Visualization:
        - X-axis: login_hour (when login occurred)
        - Y-axis: login_attempts (number of attempts)
        - Color: Blue (normal), Red (anomaly)
        - Size: Proportional to anomaly score severity

        Parameters:
        -----------
        X : pd.DataFrame
            Original features (unscaled)
        predictions : np.ndarray
            Model predictions (1 or -1)
        scores : np.ndarray
            Anomaly scores
        feature_x : str
            Feature for x-axis
        feature_y : str
            Feature for y-axis
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = OUTPUTS_DIR / "anomaly_visualization.png"

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Separate normal and anomalous points
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1

        # Calculate point sizes based on anomaly scores
        # For anomalies: more negative = larger point
        # For normal: more positive = smaller point
        sizes_normal = 50 + (scores[normal_mask] * 10)
        sizes_anomaly = 100 + (np.abs(scores[anomaly_mask]) * 50)

        # Plot normal logins
        ax.scatter(
            X.loc[normal_mask, feature_x],
            X.loc[normal_mask, feature_y],
            c=self.colors["normal"],
            s=sizes_normal,
            alpha=0.6,
            label="Normal Login",
            edgecolors="white",
            linewidth=0.5,
        )

        # Plot anomalous logins
        ax.scatter(
            X.loc[anomaly_mask, feature_x],
            X.loc[anomaly_mask, feature_y],
            c=self.colors["anomaly"],
            s=sizes_anomaly,
            alpha=0.8,
            label="Anomalous Login (ALERT!)",
            edgecolors="darkred",
            linewidth=1.5,
            marker="X",  # X marker for anomalies
        )

        # Styling
        ax.set_xlabel(
            feature_x.replace("_", " ").title(), fontsize=14, fontweight="bold"
        )
        ax.set_ylabel(
            feature_y.replace("_", " ").title(), fontsize=14, fontweight="bold"
        )
        ax.set_title(
            "Login Anomaly Detection - Isolation Forest",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Legend
        legend = ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        legend.get_frame().set_facecolor("white")

        # Add statistics box
        n_normal = normal_mask.sum()
        n_anomaly = anomaly_mask.sum()
        stats_text = f"Total Logins: {len(predictions)}\n"
        stats_text += f"Normal: {n_normal} ({n_normal/len(predictions)*100:.1f}%)\n"
        stats_text += f"Anomalies: {n_anomaly} ({n_anomaly/len(predictions)*100:.1f}%)"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Tight layout
        plt.tight_layout()

        # Save
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {filepath}")

        # Close figure to free memory
        plt.close()

    def plot_feature_distributions(self, X, predictions, filepath=None):
        """
        Compare feature distributions between normal and anomalous logins.

        Parameters:
        -----------
        X : pd.DataFrame
            Original features
        predictions : np.ndarray
            Model predictions
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = OUTPUTS_DIR / "feature_distributions.png"

        # Create subplots
        features = X.columns.tolist()
        n_features = len(features)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        # Separate data
        X_normal = X[predictions == 1]
        X_anomaly = X[predictions == -1]

        # Plot each feature
        for idx, feature in enumerate(features):
            ax = axes[idx]

            # Histograms
            ax.hist(
                X_normal[feature],
                bins=20,
                alpha=0.6,
                color=self.colors["normal"],
                label="Normal",
                density=True,
            )
            ax.hist(
                X_anomaly[feature],
                bins=20,
                alpha=0.6,
                color=self.colors["anomaly"],
                label="Anomaly",
                density=True,
            )

            ax.set_xlabel(feature.replace("_", " ").title(), fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(
                f'{feature.replace("_", " ").title()} Distribution',
                fontsize=12,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        if n_features < len(axes):
            fig.delaxes(axes[-1])

        plt.suptitle(
            "Feature Distributions: Normal vs. Anomalous Logins",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        # Save
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Feature distributions saved to {filepath}")

        plt.close()

    def plot_anomaly_scores(self, scores, predictions, filepath=None):
        """
        Plot distribution of anomaly scores.

        Parameters:
        -----------
        scores : np.ndarray
            Anomaly scores from decision_function
        predictions : np.ndarray
            Model predictions
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = OUTPUTS_DIR / "anomaly_scores.png"

        fig, ax = plt.subplots(figsize=(12, 6))

        # Separate scores
        scores_normal = scores[predictions == 1]
        scores_anomaly = scores[predictions == -1]

        # Plot distributions
        ax.hist(
            scores_normal,
            bins=50,
            alpha=0.6,
            color=self.colors["normal"],
            label="Normal",
            density=True,
        )
        ax.hist(
            scores_anomaly,
            bins=50,
            alpha=0.6,
            color=self.colors["anomaly"],
            label="Anomaly",
            density=True,
        )

        # Add threshold line
        ax.axvline(
            x=0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Decision Threshold",
            alpha=0.7,
        )

        # Styling
        ax.set_xlabel("Anomaly Score", fontsize=14, fontweight="bold")
        ax.set_ylabel("Density", fontsize=14, fontweight="bold")
        ax.set_title("Anomaly Score Distribution", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.text(
            0.02,
            0.98,
            "Negative scores → Anomalies\nPositive scores → Normal",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Anomaly scores plot saved to {filepath}")

        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, filepath=None):
        """
        Plot confusion matrix heatmap.

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = OUTPUTS_DIR / "confusion_matrix.png"

        from sklearn.metrics import confusion_matrix

        # Convert labels
        y_true_binary = np.where(y_true == "normal", 1, -1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred, labels=[1, -1])

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            cbar_kws={"label": "Count"},
            ax=ax,
            annot_kws={"size": 16},
        )

        ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
        ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

        plt.tight_layout()

        # Save
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {filepath}")

        plt.close()

    def create_dashboard(self, X, predictions, scores, y_true=None, filepath=None):
        """
        Create comprehensive dashboard with multiple visualizations.

        Parameters:
        -----------
        X : pd.DataFrame
            Original features
        predictions : np.ndarray
            Model predictions
        scores : np.ndarray
            Anomaly scores
        y_true : pd.Series, optional
            Ground truth labels
        filepath : str
            Output file path
        """
        if filepath is None:
            filepath = OUTPUTS_DIR / "dashboard.png"

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main scatter plot (top, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1

        sizes_normal = 50 + (scores[normal_mask] * 10)
        sizes_anomaly = 100 + (np.abs(scores[anomaly_mask]) * 50)

        ax1.scatter(
            X.loc[normal_mask, "login_hour"],
            X.loc[normal_mask, "login_attempts"],
            c=self.colors["normal"],
            s=sizes_normal,
            alpha=0.6,
            label="Normal",
            edgecolors="white",
            linewidth=0.5,
        )
        ax1.scatter(
            X.loc[anomaly_mask, "login_hour"],
            X.loc[anomaly_mask, "login_attempts"],
            c=self.colors["anomaly"],
            s=sizes_anomaly,
            alpha=0.8,
            label="Anomaly",
            edgecolors="darkred",
            linewidth=1.5,
            marker="X",
        )

        ax1.set_xlabel("Login Hour", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Login Attempts", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Login Anomaly Detection Dashboard", fontsize=14, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Anomaly score distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(scores[normal_mask], bins=30, alpha=0.6, color=self.colors["normal"])
        ax2.hist(scores[anomaly_mask], bins=30, alpha=0.6, color=self.colors["anomaly"])
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel("Anomaly Score", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Score Distribution", fontsize=11, fontweight="bold")

        # 3. Pie chart (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        counts = [normal_mask.sum(), anomaly_mask.sum()]
        ax3.pie(
            counts,
            labels=["Normal", "Anomaly"],
            autopct="%1.1f%%",
            colors=[self.colors["normal"], self.colors["anomaly"]],
            startangle=90,
        )
        ax3.set_title("Detection Summary", fontsize=11, fontweight="bold")

        # 4-6. Feature distributions (bottom row)
        features_to_plot = ["login_hour", "login_attempts", "ip_frequency"]
        for idx, feature in enumerate(features_to_plot):
            ax = fig.add_subplot(gs[2, idx])
            ax.hist(
                X.loc[normal_mask, feature],
                bins=15,
                alpha=0.6,
                color=self.colors["normal"],
                density=True,
            )
            ax.hist(
                X.loc[anomaly_mask, feature],
                bins=15,
                alpha=0.6,
                color=self.colors["anomaly"],
                density=True,
            )
            ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(
                f'{feature.replace("_", " ").title()}', fontsize=11, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Login Anomaly Detection Dashboard", fontsize=18, fontweight="bold", y=0.98
        )

        # Save
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Dashboard saved to {filepath}")

        plt.close()


# Example usage
if __name__ == "__main__":
    print("=== Visualizer Module Test ===")
    print("For full pipeline, run: python main.py")
