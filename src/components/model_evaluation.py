from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    model_name: str
    val_accuracy: float
    val_f1: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    classification_report_text: str
    # FIX: store predictions so callers never need to re-call predict()
    y_test_pred: Any = field(default=None, repr=False)
    y_test_score: Any = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "val_accuracy": round(self.val_accuracy, 6),
            "val_f1": round(self.val_f1, 6),
            "test_accuracy": round(self.test_accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": round(self.roc_auc, 6),
            "classification_report": self.classification_report_text,
        }


def evaluate_model(
    model_name: str,
    model: Any,
    x_val: Any,
    y_val: Any,
    x_test: Any,
    y_test: Any,
) -> ModelMetrics:
    """Evaluate a trained model on val and test sets.

    Note: models should be calibrated (e.g. CalibratedClassifierCV) so that
    predict_proba() is always available and produces meaningful probabilities.
    """
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    # All models passed here should support predict_proba (use CalibratedClassifierCV
    # for PassiveAggressiveClassifier). Fall back to decision_function only as a guard.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return ModelMetrics(
        model_name=model_name,
        val_accuracy=accuracy_score(y_val, y_val_pred),
        val_f1=f1_score(y_val, y_val_pred),
        test_accuracy=accuracy_score(y_test, y_test_pred),
        precision=precision_score(y_test, y_test_pred),
        recall=recall_score(y_test, y_test_pred),
        f1=f1_score(y_test, y_test_pred),
        roc_auc=roc_auc_score(y_test, y_score),
        classification_report_text=classification_report(
            y_test,
            y_test_pred,
            target_names=["Real", "Fake"],
        ),
        y_test_pred=y_test_pred,   # FIX: stored for reuse
        y_test_score=y_score,      # FIX: stored for reuse
    )
