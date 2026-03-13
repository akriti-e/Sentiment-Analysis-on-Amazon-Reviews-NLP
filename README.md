# Sentiment Analysis on Amazon Reviews (NLP)

A modern NLP project that classifies Amazon Alexa reviews into **Positive** or **Negative** sentiment and serves predictions through a polished **Streamlit** web app.

![App Preview](img/AppImg.png)

## Why This Project Stands Out

- End-to-end ML flow: data exploration -> preprocessing -> modeling -> deployment
- Production-oriented artifact saving (`vectorizer`, `scaler`, `classifier`)
- Interactive app with confidence score and review-quality tips
- Clean separation between training notebook and inference app

## Project Highlights

- **Dataset**: `amazon_alexa.csv`
- **Task**: Binary sentiment classification (`feedback` 0/1)
- **Text Features**: `CountVectorizer(max_features=2500)`
- **Normalization**: `MinMaxScaler`
- **Models Explored**:
  - Random Forest
  - XGBoost
  - Decision Tree
- **Deployed Model**: XGBoost (`Models/xgboost_classifier.pkl`)

## Inference Pipeline

1. User enters review text in Streamlit UI
2. Text is cleaned using regex (non-letters removed)
3. Tokens are lowercased, stopwords removed, and stemmed
4. Transformed via trained CountVectorizer
5. Scaled with trained MinMaxScaler
6. Predicted using trained XGBoost model
7. UI displays sentiment + confidence

## Tech Stack

- **Python**
- **Pandas, NumPy** for data handling
- **NLTK** for stopwords + stemming
- **Scikit-learn** for preprocessing and evaluation
- **XGBoost** for final classifier
- **Streamlit** for deployment UI
- **Matplotlib / Seaborn / WordCloud** for EDA

## Repository Structure

```text
.
├── Sentiment Analysis on Amazon Reviews.ipynb
├── app.py
├── requirements.txt
├── amazon_alexa.csv
├── Models/
│   ├── count_vectorizer.pkl
│   ├── scaler.pkl
│   └── xgboost_classifier.pkl
├── Image/
│   └── AppImg.png
└── img/
    └── AppImg.png
```

## Quick Start

### 1) Clone / open project

```bash
git clone <your-repo-url>
cd "Sentiment Analysis on Amazon Reviews - NLP"
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Launch the app

```bash
streamlit run app.py
```

The app will open in your browser automatically.

## Model Artifacts

The app loads these files at runtime:

- `Models/count_vectorizer.pkl`
- `Models/scaler.pkl`
- `Models/xgboost_classifier.pkl`

If they are missing, run the training notebook end-to-end to regenerate them.

## Evaluation Notes

The notebook includes:

- Confusion matrices
- Accuracy comparisons
- Precision, Recall, and F1 score outputs
- Classification report for model quality inspection

## UI Features

- Pastel-themed, modern interface
- Dark text for readability
- Quick sample review selector
- Confidence progress bar
- Low-confidence warning zone for uncertain predictions

## Future Improvements

- Add batch prediction via CSV upload
- Save prediction history and timestamps
- Add explainability (top contributing tokens)
- Add automated tests for preprocessing and inference
- Containerize with Docker for easy deployment

## Author

Made with Python + NLP + Streamlit for practical sentiment intelligence.
