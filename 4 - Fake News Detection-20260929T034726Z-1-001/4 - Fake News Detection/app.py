import os
import re
import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .result-box-real {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    .result-box-fake {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Natural Language Processing (NLP) Machine Learning Classifier</div>', unsafe_allow_html=True)

# Helper function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load trained artifacts with caching
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "fake_news_model.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_artifacts()

if model is None or vectorizer is None:
    st.error("⚠️ Model or TF-IDF Vectorizer file not found in `models/` directory. Please run `Fake_new_model.ipynb` first to train and save the model.")
    st.stop()

# Preset examples
examples = [
    "NASA's James Webb Telescope discovers new evidence of water on distant exoplanet",
    "SHOCKING: Secret cure for aging discovered by unknown monk, governments trying to ban it!",
    "Federal Reserve holds interest rates steady amid moderating inflation",
    "Celebrity claims drinking lemon water every morning cured all chronic diseases in 24 hours"
]

st.write("##### 💡 Try sample headlines:")
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(f"Example {i+1}: {ex[:35]}...", key=f"ex_{i}", use_container_width=True):
        st.session_state['input_text'] = ex

# Input headline
user_input = st.text_area(
    "Enter a news headline or article snippet:",
    value=st.session_state.get('input_text', ''),
    placeholder="Type or paste the news headline here...",
    height=120
)

col1, col2 = st.columns([1, 1])
predict_btn = col1.button("🔍 Check News Authenticity", type="primary", use_container_width=True)
clear_btn = col2.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state['input_text'] = ''
    st.rerun()

if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a news headline to analyze.")
    else:
        cleaned_input = clean_text(user_input)
        vec_features = vectorizer.transform([cleaned_input])
        prediction = model.predict(vec_features)[0]

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(vec_features)[0]
            fake_prob = probs[0]
            real_prob = probs[1]
        else:
            fake_prob = 1.0 if prediction == 0 else 0.0
            real_prob = 1.0 if prediction == 1 else 0.0

        is_real = (prediction == 1)
        confidence = (real_prob if is_real else fake_prob) * 100

        if is_real:
            st.markdown(f"""
            <div class="result-box-real">
                <div class="result-label" style="color: #065F46;">✅ Classified as: REAL NEWS</div>
                <div><b>Confidence Score:</b> {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-fake">
                <div class="result-label" style="color: #991B1B;">⚠️ Classified as: FAKE NEWS</div>
                <div><b>Confidence Score:</b> {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("##### Probability Breakdown")
        st.progress(real_prob)
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Real News Probability", f"{real_prob * 100:.1f}%")
        pcol2.metric("Fake News Probability", f"{fake_prob * 100:.1f}%")

        with st.expander("🛠️ View Preprocessed Text Details"):
            st.code(f"Original Text: {user_input}\nCleaned Text : {cleaned_input}", language="text")
