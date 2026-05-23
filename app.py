import json
import os

import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR   = "artifacts"
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")
METRICS_PATH    = os.path.join(ARTIFACTS_DIR, "metrics.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
    text-align: center; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,102,241,.25) 0%, transparent 70%);
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    color: #fff; margin: 0 0 .5rem; letter-spacing: -1px;
}
.hero p { color: #a5b4fc; font-size: 1.05rem; margin: 0; }

.model-badge {
    display: inline-block;
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
    color: #fff; border-radius: 30px;
    padding: .3rem 1rem; font-size: .82rem; font-weight: 500; margin-bottom: 1.2rem;
}

.result-fake {
    background: rgba(220,38,38,.12); border: 1px solid rgba(220,38,38,.4);
    border-radius: 12px; padding: 1.4rem; text-align: center;
}
.result-real {
    background: rgba(22,163,74,.12); border: 1px solid rgba(22,163,74,.4);
    border-radius: 12px; padding: 1.4rem; text-align: center;
}
.result-fake .label { font-family:'Syne',sans-serif; font-size:2rem; color:#f87171; font-weight:800; }
.result-real .label { font-family:'Syne',sans-serif; font-size:2rem; color:#4ade80; font-weight:800; }
.result-fake .conf  { color:#fca5a5; font-size:.95rem; }
.result-real .conf  { color:#86efac; font-size:.95rem; }
.conf-bar-wrap { background:#1e293b; border-radius:30px; height:10px; margin:.6rem 0; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:30px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


@st.cache_data
def load_metrics() -> dict:
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📰 Fake News Detector</h1>
  <p>NLP-powered classifier &nbsp;·&nbsp; TF-IDF + Machine Learning &nbsp;·&nbsp; Real-time predictions</p>
</div>
""", unsafe_allow_html=True)

# ── Artifact check ────────────────────────────────────────────────────────────
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.warning("⚠️ Trained artifacts not found. Run `python3 -m src.pipeline.training_pipeline` first.")
    st.stop()

metrics_data = load_metrics()
best_name    = metrics_data.get("best_model", "Unknown")

st.markdown(
    f'<div class="model-badge">🏆 Active model: {best_name} &nbsp;(selected by validation F1)</div>',
    unsafe_allow_html=True,
)

# ── Prediction ────────────────────────────────────────────────────────────────
user_input = st.text_area("Paste a news article here", height=200,
                           placeholder="e.g. The Prime Minister announced a new economic policy today...")

if st.button("🚀 Analyse Article", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing..."):
            try:
                result     = get_pipeline().predict(user_input)
                confidence = result["confidence"] * 100
                pred       = result["prediction"]

                r1, r2 = st.columns([1, 2])
                with r1:
                    if pred == 1:
                        st.markdown(
                            f'<div class="result-fake"><div class="label">🚨 FAKE</div>'
                            f'<div class="conf">{confidence:.1f}% confidence</div>'
                            f'<div class="conf-bar-wrap"><div class="conf-bar-fill" '
                            f'style="width:{confidence:.1f}%;background:#ef4444;"></div></div>'
                            f'</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="result-real"><div class="label">✅ REAL</div>'
                            f'<div class="conf">{confidence:.1f}% confidence</div>'
                            f'<div class="conf-bar-wrap"><div class="conf-bar-fill" '
                            f'style="width:{confidence:.1f}%;background:#22c55e;"></div></div>'
                            f'</div>', unsafe_allow_html=True)

                with r2:
                    word_count  = len(user_input.split())
                    clean_count = len(result["cleaned_text"].split())
                    st.metric("Words in input",        word_count)
                    st.metric("Tokens after cleaning", clean_count)
                    st.metric("Words removed",         word_count - clean_count)
                    with st.expander("🔎 Cleaned tokens used for prediction"):
                        st.code(result["cleaned_text"], language=None)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:#475569;font-size:.8rem;">'
    f'Active model: <strong style="color:#6366f1">{best_name}</strong> · '
    f'Built with scikit-learn + Streamlit</p>',
    unsafe_allow_html=True,
)
