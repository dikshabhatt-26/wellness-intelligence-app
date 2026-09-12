"""
train_model.py
Generates synthetic multi-user habit data and trains a KMeans clustering
model that groups behaviour patterns into wellness "profiles".
This is run ONCE offline; the trained model is saved to disk and loaded
by the Streamlit app at runtime (so the app itself doesn't retrain every time).

Run: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

RNG = np.random.default_rng(42)
N_SYNTHETIC_USERS = 300
DAYS_PER_USER = 30


def generate_synthetic_population(n_users=N_SYNTHETIC_USERS, days=DAYS_PER_USER):
    """
    Simulate three broad behavioural archetypes so the clustering model
    has something meaningful to learn:
      - "Balanced"      : good sleep, hydration, moderate exercise, stable mood
      - "Under-recovered": low sleep, low water, high stress/low mood
      - "Sedentary"      : okay sleep, low exercise & steps, average mood
    Each synthetic user is a noisy sample around one archetype's mean profile.
    """
    archetypes = {
        "Balanced": dict(sleep=7.5, water=2.5, mood=7.5, exercise=35, steps=8500),
        "Under-recovered": dict(sleep=5.2, water=1.4, mood=4.5, exercise=15, steps=4200),
        "Sedentary": dict(sleep=7.0, water=2.0, mood=6.0, exercise=8, steps=3000),
    }
    names = list(archetypes.keys())

    records = []
    for u in range(n_users):
        archetype = names[u % len(names)]
        base = archetypes[archetype]
        for d in range(days):
            records.append(
                dict(
                    synthetic_user=u,
                    archetype=archetype,
                    sleep_hours=max(3, RNG.normal(base["sleep"], 0.8)),
                    water_liters=max(0.3, RNG.normal(base["water"], 0.4)),
                    mood_score=int(np.clip(RNG.normal(base["mood"], 1.3), 1, 10)),
                    exercise_minutes=max(0, RNG.normal(base["exercise"], 10)),
                    steps=max(0, int(RNG.normal(base["steps"], 1500))),
                )
            )
    return pd.DataFrame(records)


def train_and_save(model_path="wellness_kmeans.joblib", scaler_path="wellness_scaler.joblib"):
    df = generate_synthetic_population()

    # Aggregate each synthetic user to their mean profile (this mirrors how we will
    # featurize a real user from their logged history at inference time).
    agg = df.groupby("synthetic_user").agg(
        sleep_hours=("sleep_hours", "mean"),
        water_liters=("water_liters", "mean"),
        mood_score=("mood_score", "mean"),
        exercise_minutes=("exercise_minutes", "mean"),
        steps=("steps", "mean"),
    ).reset_index()

    features = ["sleep_hours", "water_liters", "mood_score", "exercise_minutes", "steps"]
    X = agg[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)

    # Print cluster centers (in original units) for sanity-checking / synopsis screenshots.
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=features)
    print("Cluster centers (original units):")
    print(centers_df.round(2))
    print("\nModel and scaler saved to:", model_path, "and", scaler_path)


if __name__ == "__main__":
    train_and_save()
