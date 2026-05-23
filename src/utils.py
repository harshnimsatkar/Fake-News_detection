import json
import os
import sys
from typing import Any

import joblib

from src.exception import CustomException


def save_object(file_path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as exc:
        raise CustomException(exc, sys)


def load_object(file_path: str) -> Any:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        return joblib.load(file_path)
    except Exception as exc:
        raise CustomException(exc, sys)


def save_json(file_path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2)
    except Exception as exc:
        raise CustomException(exc, sys)
