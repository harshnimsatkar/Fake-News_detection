import os
import sys
from dataclasses import dataclass

from src.components.feature_engineering import clean_text
from src.exception import CustomException
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.joblib")
    vectorizer_path: str = os.path.join("artifacts", "vectorizer.joblib")


class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig | None = None):
        self.config = config or PredictionPipelineConfig()
        self.model = None
        self.vectorizer = None

    def _load_artifacts(self) -> None:
        """Lazy-load model and vectorizer on first call."""
        if self.model is None:
            self.model = load_object(self.config.model_path)
        if self.vectorizer is None:
            self.vectorizer = load_object(self.config.vectorizer_path)

    def predict(self, text: str) -> dict:
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")

            self._load_artifacts()

            cleaned_text = clean_text(text)   # same pipeline as training
            vectorized = self.vectorizer.transform([cleaned_text])
            prediction = int(self.model.predict(vectorized)[0])

            # FIX: all models are calibrated (CalibratedClassifierCV for PAC),
            # so predict_proba is always available and produces real probabilities.
            # The arbitrary raw_score/3 fallback is removed.
            if hasattr(self.model, "predict_proba"):
                confidence = float(self.model.predict_proba(vectorized).max())
            else:
                # Safety net only — should not be reached with current model set
                raw = abs(float(self.model.decision_function(vectorized)[0]))
                confidence = float(min(raw / 3.0, 1.0))

            return {
                "label": "FAKE" if prediction == 1 else "REAL",
                "prediction": prediction,
                "confidence": confidence,
                "cleaned_text": cleaned_text,
            }
        except Exception as exc:
            raise CustomException(exc, sys)
