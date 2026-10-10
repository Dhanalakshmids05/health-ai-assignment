
import sqlite3
from pathlib import Path

# Store the database inside question_B/level2
DB_PATH = Path(__file__).parent / "predictions.db"


def create_tables():
    """Create the predictions table if it does not exist."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction INTEGER NOT NULL,
                risk_probability REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_prediction(prediction, risk_probability):
    """Save one prediction and return its database ID."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (prediction, risk_probability)
            VALUES (?, ?)
            """,
            (prediction, risk_probability)
        )
        return cursor.lastrowid


def get_prediction_stats():
    """Return summary statistics for saved predictions."""
    with sqlite3.connect(DB_PATH) as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

        positive = connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
        ).fetchone()[0]

        negative = connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction = 0"
        ).fetchone()[0]

    return {
        "total_predictions": total,
        "positive_predictions": positive,
        "negative_predictions": negative
    }
