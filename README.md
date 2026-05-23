# 📰 Fake News Detection using Machine Learning

An end-to-end Machine Learning web application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques, deployed as an interactive web app on **Streamlit Cloud**.

🔗 **Live App:** https://newsdetection-app.streamlit.app/  
📦 **Dataset:** [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📌 Project Overview

Fake news poses a serious threat to society by spreading misinformation rapidly. This project automatically detects fake news articles using supervised ML models trained on labeled news data.

The system takes raw news text as input, runs it through a full NLP preprocessing pipeline, and predicts whether the article is **Fake** or **Real** in real time — complete with a confidence score.

---

## 🖥️ App Preview

| Section | What it shows |
|---|---|
| **Pipeline flow** | Visual step-by-step of the full ML pipeline |
| **Radar chart** | Shape of each model's strengths across all 5 metrics |
| **F1 bar chart** | Clean ranking of all models with best model highlighted |
| **Val vs Test F1 scatter** | Overfitting check — points near the diagonal = generalising well |
| **Live classifier** | Paste any article → get label + confidence bar + token inspector |

---

## 🗂️ Project Structure

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Load, merge, deduplicate raw CSVs
│   │   ├── feature_engineering.py  # Cleaning, POS-aware lemmatisation
│   │   ├── model_training.py       # Train, evaluate, select & save best model
│   │   └── model_evaluation.py     # ModelMetrics dataclass + evaluate_model()
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training run
│   │   └── prediction_pipeline.py  # Loads artifacts, predicts on new text
│   ├── exception.py                # Custom exception with file + line info
│   ├── logger.py                   # File + stdout logging
│   └── utils.py                    # save/load joblib, save JSON
├── artifacts/                      # Generated at training time (gitignored)
│   ├── Fake.csv
│   ├── True.csv
│   ├── processed_news.csv
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
├── notebook/
│   └── fake_news_detection.ipynb   # Full EDA + training notebook
├── logs/                           # Auto-generated timestamped log files
├── app.py                          # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- Modular end-to-end ML pipeline (`src/components/` + `src/pipeline/`)
- Robust NLP preprocessing: contraction expansion, POS-aware lemmatisation, stopword removal
- TF-IDF vectorisation with bigrams, `min_df` filtering, sublinear TF scaling
- Three ML models trained and compared; best selected automatically by **validation F1**
- `PassiveAggressiveClassifier` wrapped in `CalibratedClassifierCV` for real probability outputs
- Model serialisation using **joblib**
- Interactive analytics charts (radar, F1 bar, overfitting scatter)
- Confidence score + cleaned token inspector on every prediction
- Publicly deployed on Streamlit Cloud

---

## 🧠 Machine Learning Models

| Model | Notes |
|---|---|
| Multinomial Naive Bayes | Fast baseline; per-class word log-probabilities |
| Passive Aggressive Classifier | Wrapped in `CalibratedClassifierCV` for proper confidence scores |
| Logistic Regression | Strong linear baseline; interpretable coefficients |

Best model is selected automatically by **validation F1** and saved to `artifacts/model.joblib`.

---

## 📊 Evaluation Metrics

Each model is evaluated on a held-out **test set (20% of data)**:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

A **70 / 10 / 20 stratified** train / validation / test split is used. Model selection happens on the validation set so the test set is never touched during tuning.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn |
| NLP | NLTK (POS tagging, lemmatisation), TF-IDF |
| Data | pandas, numpy |
| Visualisation | matplotlib |
| Model storage | joblib |
| Web app | Streamlit |
| Deployment | Streamlit Cloud |
| Version control | Git + GitHub (Git LFS for large artifacts) |

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Fake-News_detection.git
cd Fake-News_detection
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download [Fake.csv and True.csv from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them inside the `artifacts/` folder:
```
artifacts/
├── Fake.csv
└── True.csv
```

### 5. Train the model
```bash
python3 -m src.pipeline.training_pipeline
```
This generates `artifacts/model.joblib`, `artifacts/vectorizer.joblib`, and `artifacts/metrics.json`.

### 6. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## ☁️ Deploying to Streamlit Cloud

### Step 1 — Push code to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

### Step 2 — Upload trained artifacts with Git LFS
```bash
git lfs install
git lfs track "artifacts/*.joblib"
git add .gitattributes artifacts/model.joblib artifacts/vectorizer.joblib artifacts/metrics.json
git commit -m "add trained artifacts"
git push
```

### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`
4. Click **Deploy**

Streamlit Cloud installs `requirements.txt` and launches the app automatically.

---

### 📝 Optional — Auto-retrain on first deploy (no Git LFS)

Add this block at the top of `app.py` before `get_pipeline()` is called:

```python
if not os.path.exists("artifacts/model.joblib"):
    from src.pipeline.training_pipeline import TrainingPipeline
    TrainingPipeline().run_pipeline()
```

You will also need to include `Fake.csv` and `True.csv` in the repo.

---

## 📄 License

MIT
