"""
db.py
Handles all SQLite database operations for the Wellness Intelligence app.
"""

import sqlite3
from datetime import date

DB_PATH = "wellness.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """Create tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            sleep_hours REAL,
            water_liters REAL,
            mood_score INTEGER,
            exercise_minutes REAL,
            steps INTEGER,
            UNIQUE(user_id, log_date),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
    )
    conn.commit()
    conn.close()


def get_or_create_user(name: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        cur.execute("INSERT INTO users (name) VALUES (?)", (name,))
        conn.commit()
        user_id = cur.lastrowid
    conn.close()
    return user_id


def upsert_log(user_id: int, log_date: str, sleep_hours: float, water_liters: float,
                mood_score: int, exercise_minutes: float, steps: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO logs (user_id, log_date, sleep_hours, water_liters, mood_score,
                           exercise_minutes, steps)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, log_date) DO UPDATE SET
            sleep_hours=excluded.sleep_hours,
            water_liters=excluded.water_liters,
            mood_score=excluded.mood_score,
            exercise_minutes=excluded.exercise_minutes,
            steps=excluded.steps
        """,
        (user_id, log_date, sleep_hours, water_liters, mood_score, exercise_minutes, steps),
    )
    conn.commit()
    conn.close()


def get_logs(user_id: int):
    """Return all logs for a user as a list of dict rows, ordered by date."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM logs WHERE user_id = ? ORDER BY log_date ASC", (user_id,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_all_logs(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM logs WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
