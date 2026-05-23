import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging

SEED = 42
KEEP_COLUMNS = ["text", "label"]


@dataclass
class DataIngestionConfig:
    fake_data_path: str = os.path.join("artifacts", "Fake.csv")
    true_data_path: str = os.path.join("artifacts", "True.csv")
    processed_data_path: str = os.path.join("artifacts", "processed_news.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> pd.DataFrame:
        try:
            fake_df = pd.read_csv(self.config.fake_data_path)
            true_df = pd.read_csv(self.config.true_data_path)

            for name, dataframe in {"fake": fake_df, "true": true_df}.items():
                if "text" not in dataframe.columns:
                    raise ValueError(f"{name} dataset must contain a 'text' column")

            fake_df["label"] = 1
            true_df["label"] = 0

            # FIX: keep only the columns we need — avoids saving bloated artifacts
            # with 'title', 'subject', 'date' columns that are never used downstream
            fake_df = fake_df[KEEP_COLUMNS]
            true_df = true_df[KEEP_COLUMNS]

            df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            df = df.drop_duplicates(subset=["text"]).dropna(subset=["text", "label"])
            df = df.reset_index(drop=True)

            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            df.to_csv(self.config.processed_data_path, index=False)
            logging.info("Data ingestion completed with %s rows", len(df))
            return df
        except Exception as exc:
            raise CustomException(exc, sys)
