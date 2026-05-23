import sys

from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")
            raw_dataframe = DataIngestion().initiate_data_ingestion()
            processed_dataframe = FeatureEngineering().transform(raw_dataframe)
            model, vectorizer, metrics = ModelTrainer().initiate_model_training(
                processed_dataframe
            )
            logging.info("Training pipeline completed")
            return model, vectorizer, metrics
        except Exception as exc:
            raise CustomException(exc, sys)


if __name__ == "__main__":
    _, _, model_metrics = TrainingPipeline().run_pipeline()
    print("Training completed.\n")
    for item in sorted(model_metrics, key=lambda m: m.val_f1, reverse=True):
        print(
            f"{item.model_name}: "
            f"val_f1={item.val_f1:.4f}, "
            f"test_f1={item.f1:.4f}, "
            f"test_accuracy={item.test_accuracy:.4f}, "
            f"roc_auc={item.roc_auc:.4f}"
        )
