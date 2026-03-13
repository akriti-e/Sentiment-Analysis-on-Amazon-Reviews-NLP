import pickle
import re

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


st.set_page_config(page_title="Amazon Alexa Review Sentiment", page_icon="🛒", layout="wide")


@st.cache_resource
def load_artifacts():
    with open("Models/count_vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    with open("Models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("Models/xgboost_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return cv, scaler, model


@st.cache_resource
def load_nlp_assets():
    nltk.download("stopwords")
    sw = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    return sw, stemmer


def preprocess(text, stop_words, stemmer):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)


def predict_sentiment(text, cv, scaler, model, stop_words, stemmer):
    cleaned = preprocess(text, stop_words, stemmer)
    vec = cv.transform([cleaned]).toarray()
    vec_scl = scaler.transform(vec)
    pred = model.predict(vec_scl)[0]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(vec_scl)[0][1])
    else:
        proba = None

    return int(pred), proba


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Space+Grotesk:wght@600;700&display=swap');

        :root {
            --bg-soft: #fdf7f2;
            --ink: #1e293b;
            --subtle: #475569;
            --primary: #e07a5f;
            --primary-deep: #c86447;
            --positive: #16803a;
            --negative: #b42318;
            --card: #ffffff;
            --ring: rgba(224, 122, 95, 0.25);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 0%, #f9e6dc 0%, transparent 46%),
                radial-gradient(circle at 88% 100%, #deedf7 0%, transparent 40%),
                radial-gradient(circle at 50% 30%, #eef6ea 0%, transparent 42%),
                var(--bg-soft);
            color: var(--ink);
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp .stMarkdown,
        .stApp .stCaption {
            color: var(--ink);
        }

        .hero {
            padding: 1.2rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #fdf0e8 0%, #ffffff 55%, #edf5fb 100%);
            border: 1px solid #ecdccf;
            box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
            margin-bottom: 1rem;
        }

        .hero h1 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
            color: var(--ink);
            margin: 0;
            font-size: clamp(1.6rem, 1.2rem + 1.3vw, 2.3rem);
        }

        .hero p {
            font-family: 'Manrope', sans-serif;
            color: var(--subtle);
            margin: 0.4rem 0 0;
            font-size: 1rem;
        }

        .stTextArea textarea {
            font-family: 'Manrope', sans-serif;
            border-radius: 14px;
            border: 1px solid #d4dbe8;
            box-shadow: none;
            color: var(--ink);
            background-color: #ffffff;
        }

        .stTextArea textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 0.2rem var(--ring);
        }

        .stButton button {
            background: linear-gradient(90deg, var(--primary) 0%, #ff8f64 100%);
            color: white;
            border: 0;
            font-weight: 700;
            border-radius: 12px;
            padding: 0.55rem 1.1rem;
            font-family: 'Space Grotesk', sans-serif;
        }

        .stButton button:hover {
            background: linear-gradient(90deg, var(--primary-deep) 0%, var(--primary) 100%);
        }

        .result-card {
            padding: 1rem;
            border-radius: 14px;
            background: var(--card);
            border: 1px solid #e3e8f1;
        }

        .chip {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-family: 'Manrope', sans-serif;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .chip-ok {
            background: #e7f8ee;
            color: var(--positive);
            border: 1px solid #bee9cd;
        }

        .chip-alert {
            background: #fdeceb;
            color: var(--negative);
            border: 1px solid #f5c5c2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Amazon Alexa Review Sentiment</h1>
        <p>Paste a customer review and get real-time sentiment prediction with confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.65, 1], gap="large")

with left_col:
    st.subheader("Analyze a Review")

    sample_options = {
        "Neutral placeholder": "",
        "Positive sample": "I love this speaker. The sound is clear and Alexa responds instantly.",
        "Negative sample": "Very disappointed. It disconnects often and microphone quality is poor.",
        "Mixed sample": "Good music quality but setup was difficult and app kept crashing.",
    }
    selected_sample = st.selectbox("Quick sample", list(sample_options.keys()))

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    if selected_sample != "Neutral placeholder" and st.button("Use sample text", key="use_sample"):
        st.session_state.review_text = sample_options[selected_sample]

    review = st.text_area(
        "Review text",
        value=st.session_state.review_text,
        height=220,
        placeholder="Type or paste a customer review here...",
    )
    st.session_state.review_text = review

    analyze = st.button("Analyze sentiment", use_container_width=True)

with right_col:
    st.subheader("Model Info")
    st.markdown(
        """
        <span class="chip chip-ok">XGBoost Classifier</span>
        <span class="chip chip-ok">Count Vectorizer</span>
        <span class="chip chip-ok">MinMax Scaling</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Pipeline: clean text -> stem/stopword filter -> vectorize -> scale -> predict")
    st.markdown("---")
    st.subheader("Tips")
    st.markdown(
        """
        - Use complete sentences for better confidence.
        - Avoid pasting multiple unrelated reviews together.
        - Confidence is model-estimated probability, not certainty.
        """
    )

if analyze:
    if not review.strip():
        st.warning("Please enter review text.")
    else:
        cv, scaler, model = load_artifacts()
        stop_words, stemmer = load_nlp_assets()
        pred, proba = predict_sentiment(review, cv, scaler, model, stop_words, stemmer)

        st.markdown("### Prediction")
        if pred == 1:
            st.success("Positive sentiment detected")
        else:
            st.error("Negative sentiment detected")

        if proba is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f"Positive confidence: {proba:.2%}")
            st.progress(int(max(0.0, min(1.0, proba)) * 100))
            if 0.4 <= proba <= 0.6:
                st.markdown(
                    '<span class="chip chip-alert">Low-confidence zone: consider manual review</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
