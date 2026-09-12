"""
app.py
AI-Powered Personal Wellness and Habit Intelligence
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

import db
import ml_models as ml

st.set_page_config(page_title="Wellness Intelligence", page_icon="🌿", layout="wide")
db.init_db()

# ---------------------------------------------------------------------------
# Sidebar: simple name-based "login" (no auth needed for a demo/minor project)
# ---------------------------------------------------------------------------
st.sidebar.title("🌿 Wellness Intelligence")
st.sidebar.caption("AI-Powered Personal Wellness & Habit Intelligence")

user_name = st.sidebar.text_input("Enter your name to continue", value="")

if not user_name.strip():
    st.title("Welcome 👋")
    st.write(
        "Enter your name in the sidebar to start logging your habits and "
        "unlock AI-based recommendations and forecasts."
    )
    st.stop()

user_id = db.get_or_create_user(user_name.strip())

page = st.sidebar.radio(
    "Navigate",
    ["Log Habit", "Dashboard", "Insights & Recommendations", "Predictions"],
)

st.sidebar.divider()
if st.sidebar.button("⚠️ Clear my data"):
    db.delete_all_logs(user_id)
    st.sidebar.success("All logs cleared.")

logs = db.get_logs(user_id)
logs_df = pd.DataFrame(logs)
if not logs_df.empty:
    logs_df["log_date"] = pd.to_datetime(logs_df["log_date"])
    logs_df = logs_df.sort_values("log_date")

# ---------------------------------------------------------------------------
# PAGE 1: Log Habit
# ---------------------------------------------------------------------------
if page == "Log Habit":
    st.title("📝 Log Today's Habits")
    st.write("Enter your daily wellness data. You can also backfill a past date.")

    with st.form("log_form"):
        col1, col2 = st.columns(2)
        with col1:
            log_date = st.date_input("Date", value=date.today())
            sleep_hours = st.slider("Sleep (hours)", 0.0, 12.0, 7.0, 0.25)
            water_liters = st.slider("Water intake (liters)", 0.0, 5.0, 2.0, 0.1)
        with col2:
            mood_score = st.slider("Mood (1 = low, 10 = great)", 1, 10, 6)
            exercise_minutes = st.slider("Exercise (minutes)", 0, 120, 20, 5)
            steps = st.number_input("Steps", min_value=0, max_value=50000, value=5000, step=500)

        submitted = st.form_submit_button("Save Entry")
        if submitted:
            db.upsert_log(
                user_id, str(log_date), sleep_hours, water_liters,
                mood_score, exercise_minutes, int(steps),
            )
            st.success(f"Saved entry for {log_date}.")
            st.rerun()

    if not logs_df.empty:
        st.subheader("Your recent entries")
        display_df = logs_df[["log_date", "sleep_hours", "water_liters", "mood_score",
                               "exercise_minutes", "steps"]].sort_values("log_date", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE 2: Dashboard
# ---------------------------------------------------------------------------
elif page == "Dashboard":
    st.title("📊 Dashboard")

    if logs_df.empty:
        st.info("No data yet — log a few days of habits first.")
    else:
        score = ml.compute_wellness_score(logs_df.tail(7))
        streak = ml.compute_streak(logs_df["log_date"].dt.strftime("%Y-%m-%d").tolist())

        c1, c2, c3 = st.columns(3)
        c1.metric("Wellness Score (last 7 days)", f"{score} / 100")
        c2.metric("Current Logging Streak", f"{streak} day(s)")
        c3.metric("Total Entries", len(logs_df))

        st.divider()
        metric_choice = st.selectbox(
            "Metric to chart",
            ["sleep_hours", "water_liters", "mood_score", "exercise_minutes", "steps"],
        )
        fig = px.line(logs_df, x="log_date", y=metric_choice, markers=True,
                       title=f"{metric_choice.replace('_', ' ').title()} over time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("All metrics (normalized view)")
        norm_df = logs_df.copy()
        for col, maxv in [("sleep_hours", 10), ("water_liters", 3), ("mood_score", 10),
                           ("exercise_minutes", 60), ("steps", 10000)]:
            norm_df[col] = norm_df[col] / maxv
        fig2 = px.line(norm_df, x="log_date",
                        y=["sleep_hours", "water_liters", "mood_score", "exercise_minutes", "steps"],
                        title="All habits (normalized 0-1 for comparison)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3: Insights & Recommendations (Cluster-based ML)
# ---------------------------------------------------------------------------
elif page == "Insights & Recommendations":
    st.title("🧠 AI Insights & Recommendations")

    if logs_df.empty or len(logs_df) < 3:
        st.info("Log at least 3 days of data to unlock your AI profile and recommendations.")
    else:
        recent = logs_df.tail(14)  # use up to last 2 weeks
        rec = ml.get_recommendation(recent)

        st.subheader(f"Your Behavioural Profile: **{rec['label']}**")
        st.write(rec["summary"])

        st.subheader("Personalized Tips")
        for tip in rec["tips"]:
            st.markdown(f"- {tip}")

        st.divider()
        st.caption(
            "This profile is generated using a KMeans clustering model trained on "
            "simulated behavioural archetypes (Balanced, Under-recovered, Sedentary). "
            "Your last 14 days of logged habits are averaged and matched to the nearest cluster."
        )

# ---------------------------------------------------------------------------
# PAGE 4: Predictions (Forecasting)
# ---------------------------------------------------------------------------
elif page == "Predictions":
    st.title("🔮 Predictive Analytics")

    if logs_df.empty or len(logs_df) < 3:
        st.info("Log at least 3 days of data to generate a forecast.")
    else:
        metric_choice = st.selectbox(
            "Metric to forecast",
            ["sleep_hours", "water_liters", "mood_score", "exercise_minutes", "steps"],
            key="forecast_metric",
        )
        days_ahead = st.slider("Days to forecast", 3, 14, 7)

        dates = logs_df["log_date"].tolist()
        values = logs_df[metric_choice].tolist()
        f_dates, f_values = ml.forecast_metric(dates, values, days_ahead)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=f_dates, y=f_values, mode="lines+markers", name="Forecast",
                                  line=dict(dash="dash")))
        fig.update_layout(title=f"{metric_choice.replace('_', ' ').title()} — Actual vs Forecast")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Forecast uses a least-squares linear trend fit on your historical logs. "
            "With more data, this can be swapped for a more advanced time-series model "
            "(e.g., ARIMA or Prophet) without changing the rest of the app."
        )
