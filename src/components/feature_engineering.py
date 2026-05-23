import re
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.exception import CustomException
from src.logger import logging


CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "didn't": "did not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "they're": "they are",
    "they've": "they have",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
}


def _load_stop_words() -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except Exception:
        logging.warning("nltk stopwords unavailable — falling back to sklearn ENGLISH_STOP_WORDS")
        return set(ENGLISH_STOP_WORDS)


def _load_lemmatizer():
    """Return (lemmatizer, pos_tag_fn) or (None, None) if nltk unavailable."""
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer

        for resource in ("wordnet", "omw-1.4", "averaged_perceptron_tagger"):
            try:
                nltk.data.find(f"corpora/{resource}" if resource != "averaged_perceptron_tagger"
                               else f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # warm-up check

        from nltk import pos_tag

        # Map Penn Treebank POS → WordNet POS
        def get_wordnet_pos(tag: str):
            if tag.startswith("J"):
                return wordnet.ADJ
            if tag.startswith("V"):
                return wordnet.VERB
            if tag.startswith("R"):
                return wordnet.ADV
            return wordnet.NOUN  # default

        def pos_aware_lemmatize(tokens: list[str]) -> list[str]:
            """FIX: POS-aware lemmatisation so verbs/adjectives/adverbs are handled correctly."""
            tagged = pos_tag(tokens)
            return [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged]

        return pos_aware_lemmatize

    except Exception:
        logging.warning(
            "nltk lemmatizer/pos_tagger unavailable — lemmatisation will be skipped"
        )
        return None


STOP_WORDS = _load_stop_words()
_LEMMATIZE: Optional[callable] = _load_lemmatizer()   # None if nltk unavailable


def expand_contractions(text: str) -> str:
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expanded, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    if _LEMMATIZE is not None:
        tokens = _LEMMATIZE(tokens)   # FIX: POS-aware lemmatisation
    return " ".join(tokens)


@dataclass
class FeatureEngineeringConfig:
    text_column: str = "text"
    label_column: str = "label"
    clean_text_column: str = "clean_text"


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig | None = None):
        self.config = config or FeatureEngineeringConfig()

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            required_columns: list[str] = [
                self.config.text_column,
                self.config.label_column,
            ]
            missing_columns = [c for c in required_columns if c not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")

            df = dataframe.copy()
            df[self.config.clean_text_column] = df[self.config.text_column].apply(clean_text)
            df = df[df[self.config.clean_text_column].str.len() > 0].reset_index(drop=True)
            logging.info("Feature engineering completed with %s rows", len(df))
            return df
        except Exception as exc:
            raise CustomException(exc, sys)
