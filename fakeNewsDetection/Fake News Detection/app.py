import os
import re
import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

# Clean, minimal white background styling
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #111111;
    }
    textarea {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Fake News Detector")
st.write("Testing interface to check whether a news headline is Real or Fake.")

# Locate model artifacts robustly
def get_model_paths():
    candidates = [
        os.path.join(os.path.dirname(__file__), "models"),
        os.path.join(os.path.dirname(__file__), "4 - Fake News Detection-20260929T034726Z-1-001", "4 - Fake News Detection", "models"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "4 - Fake News Detection-20260929T034726Z-1-001", "4 - Fake News Detection", "models")
    ]
    for d in candidates:
        m = os.path.join(d, "fake_news_model.pkl")
        v = os.path.join(d, "tfidf_vectorizer.pkl")
        if os.path.exists(m) and os.path.exists(v):
            return m, v
    return None, None

@st.cache_resource
def load_artifacts():
    m_path, v_path = get_model_paths()
    if not m_path or not v_path:
        return None, None
    model = joblib.load(m_path)
    vectorizer = joblib.load(v_path)
    return model, vectorizer

model, vectorizer = load_artifacts()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if model is None or vectorizer is None:
    st.error("Model files not found. Please verify models/fake_news_model.pkl exists.")
    st.stop()

# Text input for headline
news_text = st.text_area(
    "Enter News Headline:",
    height=130,
    placeholder="Type or paste news text here..."
)

if st.button("Check News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter a headline first.")
    else:
        cleaned = clean_text(news_text)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            conf = max(probs) * 100
        else:
            conf = None

        if pred == 1:
            if conf is not None:
                st.success(f"Result: Real News (Confidence: {conf:.1f}%)")
            else:
                st.success("Result: Real News")
        else:
            if conf is not None:
                st.error(f"Result: Fake News (Confidence: {conf:.1f}%)")
            else:
                st.error("Result: Fake News")
