import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Recipe Rating Predictor", page_icon="🍽️", layout="centered")

# ── Minimal dark CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body, .stApp, [class*="css"] {
    background-color: #0f1117 !important;
    color: #e2e8f0;
}

/* Section labels */
h1 { font-size: 1.4rem !important; font-weight: 600; color: #f1f5f9 !important; }

/* Inputs */
input[type="number"] {
    background: #1a1d27 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 8px !important;
}

label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Predict button */
div.stButton > button {
    background: #ff6b35;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    margin-top: 1rem;
    cursor: pointer;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #e55a25; }

/* Result box */
.result {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.big-score { font-size: 3.5rem; font-weight: 700; color: #ff6b35; }
.stars     { font-size: 1.8rem; margin: 0.3rem 0; }
.label     { font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem; }

/* Divider */
hr { border-color: #2d3148 !important; }

/* Sidebar */
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in [
        os.path.join(root, "notebooks", "recipe_rating_pipline.pkl"),
        os.path.join(root, "model.pkl"),
        r"D:\Training\recipe_rating_ML_project\notebooks\recipe_rating_pipline.pkl"
    ]:
        if os.path.exists(path):
            return joblib.load(path), None
    return None, "Pipeline not found."

pipeline, load_error = load_pipeline()

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("## 🍽️ Recipe Rating Predictor")
st.markdown("<p style='color:#64748b; font-size:0.9rem; margin-top:-0.8rem;'>Enter comment details to predict the star rating.</p>", unsafe_allow_html=True)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    recipe_number    = st.number_input("Recipe Number",   min_value=1,   max_value=100,    value=1,       step=1)
    thumbs_up        = st.number_input("Thumbs Up",       min_value=0,   max_value=10000,  value=10)
    reply_count      = st.number_input("Reply Count",     min_value=0,   max_value=500,    value=2)

with c2:
    user_reputation  = st.number_input("User Reputation", min_value=0,   max_value=100000, value=500,     step=50)
    thumbs_down      = st.number_input("Thumbs Down",     min_value=0,   max_value=10000,  value=1)
    best_score       = st.number_input("Best Score",      min_value=0,   max_value=100000, value=20)

creation_timestamp = st.number_input("Creation Timestamp (Unix)", min_value=0, max_value=9999999999, value=1609459200)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Rating"):
    if pipeline is None:
        st.error(load_error)
    else:
        input_df = pd.DataFrame({
            "RecipeNumber":      [recipe_number],
            "UserReputation":    [user_reputation],
            "ReplyCount":        [reply_count],
            "ThumbsUpCount":     [thumbs_up],
            "ThumbsDownCount":   [thumbs_down],
            "BestScore":         [best_score],
            "CreationTimestamp": [creation_timestamp],
        })

        try:
            pred       = float(np.clip(pipeline.predict(input_df)[0], 1.0, 5.0))
            label      = ("Excellent 🏆" if pred >= 4.5 else
                          "Good 👍"      if pred >= 3.5 else
                          "Average 😐"  if pred >= 2.5 else
                          "Low Rating ⚠️")

            st.markdown(f"""
            <div class="result">
                <div class="big-score">{round(pred, 2)} <span style="font-size:1.5rem;color:#64748b;">/ 5</span></div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
