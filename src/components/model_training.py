import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.components.model_evaluation import ModelMetrics, evaluate_model
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object


SEED = 42


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.joblib")
    vectorizer_path: str = os.path.join("artifacts", "vectorizer.joblib")
    metrics_path: str = os.path.join("artifacts", "metrics.json")
    clean_text_column: str = "clean_text"
    label_column: str = "label"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    @staticmethod
    def _candidate_models() -> dict:
        pac_base = PassiveAggressiveClassifier(C=0.1, max_iter=1000, random_state=SEED)
        return {
            "Naive Bayes": MultinomialNB(alpha=0.1),
            # FIX: wrap PAC in CalibratedClassifierCV so it exposes predict_proba()
            # and produces meaningful confidence scores (not raw decision values)
            "Passive Aggressive": CalibratedClassifierCV(pac_base, cv=3),
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=SEED,
                solver="lbfgs",
            ),
        }

    @staticmethod
    def _build_vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            stop_words="english",
            max_df=0.70,
            min_df=3,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=60_000,
        )

    def initiate_model_training(
        self, dataframe: pd.DataFrame
    ) -> tuple[object, TfidfVectorizer, list[ModelMetrics]]:
        try:
            x = dataframe[self.config.clean_text_column]
            y = dataframe[self.config.label_column]

            # 70 / 10 / 20 stratified split
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                x, y, test_size=0.20, random_state=SEED, stratify=y
            )
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val, y_train_val, test_size=0.125, random_state=SEED, stratify=y_train_val
            )

            vectorizer = self._build_vectorizer()
            x_train_vec = vectorizer.fit_transform(x_train)  # fit on train only — no leakage
            x_val_vec = vectorizer.transform(x_val)
            x_test_vec = vectorizer.transform(x_test)

            trained_models: dict = {}
            metrics: list[ModelMetrics] = []

            for model_name, model in self._candidate_models().items():
                logging.info("Training model: %s", model_name)
                model.fit(x_train_vec, y_train)
                trained_models[model_name] = model
                result = evaluate_model(
                    model_name, model, x_val_vec, y_val, x_test_vec, y_test
                )
                metrics.append(result)
                logging.info(
                    "%s — val_f1=%.4f  test_f1=%.4f  roc_auc=%.4f",
                    model_name, result.val_f1, result.f1, result.roc_auc,
                )

            # Select best model by val_f1, break ties with val_accuracy
            best_metrics = sorted(
                metrics,
                key=lambda m: (m.val_f1, m.val_accuracy),
                reverse=True,
            )[0]
            best_model = trained_models[best_metrics.model_name]

            save_object(self.config.model_path, best_model)
            save_object(self.config.vectorizer_path, vectorizer)
            save_json(
                self.config.metrics_path,
                {
                    "best_model": best_metrics.model_name,
                    "selection_metric": "validation_f1",
                    "models": [m.as_dict() for m in metrics],
                },
            )

            logging.info(
                "Saved best model '%s' → %s", best_metrics.model_name, self.config.model_path
            )
            return best_model, vectorizer, metrics
        except Exception as exc:
            raise CustomException(exc, sys)
