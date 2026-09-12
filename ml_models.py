"""
ml_models.py
Two pieces of "emerging tech" intelligence for the app:

1. Recommendation Engine
   - Loads the pre-trained KMeans model (trained in train_model.py).
   - Given a user's recent averaged habit profile, assigns them to a
     behavioural cluster and returns a human-readable profile + tips.

2. Predictive Analytics
   - Given a user's time-series of logs for a given metric, fits a simple
     linear trend (least-squares) and forecasts the next N days.
   - Also computes a composite "Wellness Score" and current streaks.
"""

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "wellness_kmeans.joblib"
SCALER_PATH = "wellness_scaler.joblib"

FEATURES = ["sleep_hours", "water_liters", "mood_score", "exercise_minutes", "steps"]

# Human-readable labels + advice mapped from the *learned* cluster centers.
# NOTE: cluster index -> meaning was determined by inspecting centers after
# training (see train_model.py output). If you retrain, re-check this mapping.
CLUSTER_PROFILES = {
    0: {
        "label": "Under-recovered",
        "summary": "Your sleep and hydration are running low, and mood/exercise are trailing behind. "
                    "Small recovery-focused changes will likely have the biggest impact right now.",
        "tips": [
            "Aim for a consistent sleep window \u2014 even +45 minutes/night compounds fast.",
            "Keep a water bottle at your desk; sip on a schedule rather than waiting to feel thirsty.",
            "A 10-minute walk after meals can lift both mood and step count without needing 'real' workout time.",
        ],
    },
    1: {
        "label": "Balanced",
        "summary": "You're maintaining strong, well-rounded habits across sleep, hydration, mood and activity. "
                    "The focus now is consistency, not overhaul.",
        "tips": [
            "Protect your current sleep schedule \u2014 it's your strongest asset.",
            "Consider adding light strength training to diversify your exercise routine.",
            "Track mood dips around specific days/weeks to catch early signs of burnout.",
        ],
    },
    2: {
        "label": "Sedentary",
        "summary": "Sleep is reasonable, but activity levels (exercise & steps) are low. "
                    "Movement is the highest-leverage area to improve.",
        "tips": [
            "Start with a small, non-negotiable daily walk (10\u201315 minutes) rather than long gym sessions.",
            "Try 'movement snacking' \u2014 short bursts of activity between tasks/classes.",
            "Pair exercise with something you already enjoy (music, a podcast) to lower the activation cost.",
        ],
    },
}


def _load_model_and_scaler():
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return kmeans, scaler


def get_recommendation(recent_logs_df: pd.DataFrame):
    """
    recent_logs_df: DataFrame with columns in FEATURES, one row per day.
    Returns dict with cluster label, summary, and tips.
    """
    if recent_logs_df.empty:
        return None

    profile = recent_logs_df[FEATURES].mean().values.reshape(1, -1)
    kmeans, scaler = _load_model_and_scaler()
    profile_scaled = scaler.transform(profile)
    cluster_id = int(kmeans.predict(profile_scaled)[0])
    return CLUSTER_PROFILES.get(cluster_id, CLUSTER_PROFILES[1])


def compute_wellness_score(recent_logs_df: pd.DataFrame) -> float:
    """
    Composite 0-100 wellness score from the last available logs.
    Weights are a simple, explainable heuristic (not the ML model) --
    useful as an always-available fallback metric.
    """
    if recent_logs_df.empty:
        return 0.0

    row = recent_logs_df[FEATURES].mean()

    sleep_score = np.clip(row["sleep_hours"] / 8.0, 0, 1) * 25
    water_score = np.clip(row["water_liters"] / 3.0, 0, 1) * 20
    mood_score = np.clip(row["mood_score"] / 10.0, 0, 1) * 25
    exercise_score = np.clip(row["exercise_minutes"] / 30.0, 0, 1) * 15
    steps_score = np.clip(row["steps"] / 8000.0, 0, 1) * 15

    total = sleep_score + water_score + mood_score + exercise_score + steps_score
    return round(float(total), 1)


def forecast_metric(dates: list, values: list, days_ahead: int = 7):
    """
    Simple linear trend forecast using least-squares fit.
    dates: list of pandas Timestamps (or date strings), ascending order.
    values: list of numeric values, same length as dates.
    Returns (forecast_dates, forecast_values).
    """
    if len(values) < 3:
        return [], []

    x = np.arange(len(values))
    y = np.array(values, dtype=float)

    # Fit a straight line y = m*x + c
    m, c = np.polyfit(x, y, 1)

    future_x = np.arange(len(values), len(values) + days_ahead)
    future_y = m * future_x + c
    future_y = np.clip(future_y, 0, None)  # habits can't go negative

    last_date = pd.to_datetime(dates[-1])
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(days_ahead)]

    return future_dates, future_y.tolist()


def compute_streak(dates: list) -> int:
    """Count consecutive days (ending today or the most recent log) with an entry."""
    if not dates:
        return 0
    parsed = sorted(pd.to_datetime(dates))
    streak = 1
    for i in range(len(parsed) - 1, 0, -1):
        if (parsed[i] - parsed[i - 1]).days == 1:
            streak += 1
        else:
            break
    return streak
