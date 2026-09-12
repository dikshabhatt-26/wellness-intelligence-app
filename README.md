# AI-Powered Personal Wellness and Habit Intelligence

Minor Project — MCA 2nd Year
Team: Diksha Bhatt & Sachin Parihar

## What this app does
A Streamlit web app where a user logs daily wellness habits (sleep, water intake,
mood, exercise, steps) and receives:

1. **Dashboard** — trends, a composite Wellness Score, and logging streaks.
2. **AI Insights & Recommendations** — a KMeans clustering model (trained on
   simulated behavioural archetypes: Balanced / Under-recovered / Sedentary)
   matches the user's recent habits to a profile and gives personalized tips.
3. **Predictive Analytics** — a linear trend forecast projects any habit metric
   forward by 3–14 days.

## Project structure
- `app.py` — main Streamlit application (all 4 pages/screens)
- `db.py` — SQLite persistence layer (users + daily logs)
- `ml_models.py` — recommendation engine + forecasting logic (used by app.py)
- `train_model.py` — offline script that generates synthetic training data and
  trains/saves the KMeans clustering model (`wellness_kmeans.joblib`,
  `wellness_scaler.joblib`). Already run once — re-run only if you want to retrain.
- `requirements.txt` — Python dependencies

## How to run
```bash
pip install -r requirements.txt
python train_model.py      # only needed once, model files are already included
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Tech stack
Python, Streamlit, scikit-learn (KMeans), NumPy/Pandas, Plotly, SQLite

## Notes for the synopsis report
- Emerging tech components: unsupervised ML (clustering) for behavioural
  profiling, and predictive analytics (trend forecasting) — both are highlighted
  in the app's "Insights" and "Predictions" pages, useful for screenshots.
- The clustering model is trained offline on simulated data representing three
  behavioural archetypes so it generalizes reasonably even with few real logs.
