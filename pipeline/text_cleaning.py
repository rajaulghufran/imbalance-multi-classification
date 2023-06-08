import re
from datetime import datetime
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import strip_accents_ascii

from .data.emoticons import EMOTICON_PATTERNS
from .data.sub_patterns import SUB_PATTERNS_1, SUB_PATTERNS_2

class TextCleaning(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[str], y=None) -> List[str]:
        if self.verbose > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: TEXT CLEANING')

        X_cleaned: List[str] = []

        for string in X:
            # normalize unicode to ascii
            # remove emoji
            string = strip_accents_ascii(string)

            for pattern, repl in SUB_PATTERNS_1:
                string = re.sub(pattern, repl, string, flags=re.IGNORECASE)

            # remove emoticons
            string = EMOTICON_PATTERNS.sub("", string)

            for pattern, repl in SUB_PATTERNS_2:
                string = re.sub(pattern, repl, string, flags=re.IGNORECASE)

            X_cleaned.append(string.strip())

        return X_cleaned
