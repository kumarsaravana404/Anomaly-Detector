import os
import psycopg2
from datetime import datetime
import json
from src.logger import logger

# Get Database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db_connection():
    """Establish connection to PostgreSQL database."""
    if not DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def init_db():
    """Create logs table if it doesn't exist."""
    conn = get_db_connection()
    if not conn:
        logger.warning("No database connection. Logging to DB is disabled.")
        return

    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS access_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_frequency INTEGER,
                login_hour INTEGER,
                login_attempts INTEGER,
                device_type INTEGER,
                login_success INTEGER,
                prediction VARCHAR(20),
                risk_level VARCHAR(20),
                anomaly_score FLOAT
            );
        """
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")


def insert_log(data, prediction, risk_level, score):
    """Insert a new log entry into the database."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO access_logs 
            (ip_frequency, login_hour, login_attempts, device_type, login_success, prediction, risk_level, anomaly_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                data.get("ip_frequency"),
                data.get("login_hour"),
                data.get("login_attempts"),
                data.get("device_type"),
                data.get("login_success"),
                prediction,
                risk_level,
                float(score),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log to database: {e}")
