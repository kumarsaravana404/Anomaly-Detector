import os
import psycopg2


def get_connection():
    try:
        # Read DATABASE_URL from environment variables
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set")

        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def init_db():
    """
    Initialize the database.
    Since the table is already created, this function is a placeholder
    to ensure compatibility with existing app imports.
    """
    pass


def insert_log(data, risk_score, risk_level):
    conn = None
    try:
        conn = get_connection()
        if conn is None:
            return

        with conn:
            with conn.cursor() as cur:
                # insert into table login_logs
                query = """
                    INSERT INTO login_logs (
                        failed_attempts,
                        login_hour,
                        ip_risk_score,
                        device_type,
                        risk_score,
                        risk_level
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """

                # Map data dictionary keys to table columns
                # Assuming 'login_attempts' maps to 'failed_attempts'
                # Assuming 'ip_frequency' maps to 'ip_risk_score' based on app context
                values = (
                    data.get("login_attempts", 0),
                    data.get("login_hour", 0),
                    data.get("ip_frequency", 0),
                    data.get("device_type", 0),
                    risk_score,
                    risk_level,
                )

                cur.execute(query, values)

        # Connection uses transaction context above, but we still need to close it
    except Exception as e:
        print(f"Error inserting log: {e}")
    finally:
        if conn:
            conn.close()
