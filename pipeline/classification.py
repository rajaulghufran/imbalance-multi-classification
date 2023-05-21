import pickle
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from time import time
from typing import Dict, List, Literal, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearnex import config_context
from sklearnex.model_selection import train_test_split
from sklearnex.svm import SVC

from .document_transformer import DocumentTransformer
from .pos_filter import POSFilter
from .stopword_removal import StopWordRemoval
from .text_cleaning import TextCleaning
from .tokenize_mwt_pos_lemma import TokenizeMWTPOSLemma
# from .usm_ndarray_transformer import USMndarrayTransformer

def fun(arg):
    return arg

class Classification:
    def __init__(self) -> None:
        cachedir = mkdtemp()

        self.text_preprocessing_pipeline: Pipeline = Pipeline(
            [
                ("text_cleaning", TextCleaning()),
                ("tokenize_mwt_pos_lemma", TokenizeMWTPOSLemma())
            ],
            memory=cachedir
        )

        self.feature_selection_pipeline: Pipeline = Pipeline(
            [
                ("pos_filter", POSFilter()),
                ("stopword_removal", StopWordRemoval()),
                ("document_transformer", DocumentTransformer())
            ],
            memory=cachedir
        )

        tfidfvectorizer_hyperparameters = {
            "encoding": "ascii",
            "decode_error": "ignore",
            "strip_accents": "ascii",
            "preprocessor": fun,
            "tokenizer": fun,
            "analyzer": "word",
            "token_pattern": None,
        }

        svc_hyperparameters = {
            "class_weight": "balanced",
            "decision_function_shape": "ovo",
            "random_state": 42
        }

        self.classification_pipeline: Pipeline = Pipeline(
            [
                ("tfidfvectorizer", TfidfVectorizer(**tfidfvectorizer_hyperparameters)),
                # ("usm_ndarray_transformer", USMndarrayTransformer()),
                ("svc", SVC(**svc_hyperparameters)),
            ],
            memory=cachedir
        )

    def get_model_attrs(self):
        pos_filter = self.feature_selection_pipeline.named_steps["pos_filter"]
        stopword_removal = self.feature_selection_pipeline.named_steps["stopword_removal"]
        tfidfvectorizer = self.classification_pipeline.named_steps["tfidfvectorizer"]
        svc = self.classification_pipeline.named_steps["svc"]

        attrs = {
            "pos_filter__pos": pos_filter.pos,
            "stopword_removal__stopwords": set(stopword_removal.stopwords),
            "tfidfvectorizer__ngram_range": tfidfvectorizer.ngram_range,
            "tfidfvectorizer__min_df": tfidfvectorizer.min_df,
            "tfidfvectorizer__max_df": tfidfvectorizer.max_df,
            "tfidfvectorizer__stop_words": tfidfvectorizer.stop_words_,
            "tfidfvectorizer__vocabulary": tfidfvectorizer.vocabulary_,
            "svc__kernel": svc.kernel,
            "svc__C": svc.C
        }

        if svc.kernel == "rbf":
            attrs["svc__gamma"] = svc.gamma

        return attrs

    def train_test_split(
            self,
            X: Iterable,
            y: Iterable,
            test_size: float = .2
    ) -> Tuple[Iterable, Iterable, Iterable, Iterable]:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    def tuning(
        self,
        X_train: Iterable,
        y_train: Iterable,
        param_grid: Union[None, List[Dict[str, any]]] = None,
        n_jobs: int = 1,
        verbose: Literal[1, 2, 3] = 1
    ) -> Tuple[GridSearchCV, float]:
        if param_grid is None:
            param_grid = [
                {
                    "tfidfvectorizer__ngram_range": ((1, 1), (1, 2)),
                    "tfidfvectorizer__min_df": (1, 3, 5, 10),
                    "tfidfvectorizer__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
                    "svc__kernel": ("linear",),
                    "svc__C": (0.01, 0.1, 1, 10, 100, 1000, 10000)
                },
                {
                    "tfidfvectorizer__ngram_range": ((1, 1), (1, 2)),
                    "tfidfvectorizer__min_df": (1, 3, 5, 10),
                    "tfidfvectorizer__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
                    "svc__kernel": ("rbf",),
                    "svc__C": (0.01, 0.1, 1, 10, 100, 1000, 10000),
                    "svc__gamma": (0.0001, 0.001, 0.01, 0.1, 1)
                }
            ]
            
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: HYPER-PARAMETERS TUNING')

        grid_search = GridSearchCV(
            estimator=self.classification_pipeline,
            param_grid=param_grid,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=n_jobs,
            cv=StratifiedShuffleSplit(n_splits=5, test_size=.2, random_state=42),
            verbose=verbose
        )

        t0 = time()

        with config_context(target_offload="auto", allow_fallback_to_host=True):
            grid_search.fit(X_train, y_train)

        return (grid_search, time() - t0)

    def train(self, X_train: Iterable, y_train: Iterable) -> None:
        X_train = self.text_preprocessing_pipeline.transform(X_train)
        X_train = self.feature_selection_pipeline.transform(X_train)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TRAINING')
        with config_context(target_offload="auto", allow_fallback_to_host=True):
            self.classification_pipeline.fit(X_train, y_train)

    def train_preprocessed(self, X_train: Iterable, y_train: Iterable) -> None:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TRAINING')

        with config_context(target_offload="auto", allow_fallback_to_host=True):
            self.classification_pipeline.fit(X_train, y_train)
    
    def test(self, X_test: Iterable) -> Iterable[int]:
        X_test = self.text_preprocessing_pipeline.transform(X_test)
        X_test = self.feature_selection_pipeline.transform(X_test)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TESTINIG')
        return self.classification_pipeline.predict(X_test)
    
    def test_preprocessed(self, X_test: Iterable) -> Iterable[int]:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TESTINIG')
        return self.classification_pipeline.predict(X_test)
    
    def score(self, y_test: Iterable, y_pred: Iterable):
        return (
            accuracy_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred)
        )
    
    def is_fitted(self) -> bool:
        try:
            check_is_fitted(self.classification_pipeline.named_steps["tfidfvectorizer"], ["vocabulary_"])
            check_is_fitted(self.classification_pipeline.named_steps["svc"], ["classes_"])
            return True

        except NotFittedError:
            return False
    
    def to_bytes(self) -> bytes:
        return pickle.dumps((self.feature_selection_pipeline, self.classification_pipeline))
    
    def to_disk(self, dirpath: Path) -> None:
        with open(dirpath / f'model.{datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")}.pickle', "wb") as f:
            pickle.dump((self.feature_selection_pipeline, self.classification_pipeline), f)

    def from_bytes(self, bytes_: bytes) -> None:
        self.feature_selection_pipeline, self.classification_pipeline = pickle.loads(bytes_)

    def from_disk(self, filepath: Path) -> None:
        with open(filepath, "rb") as f:
            self.feature_selection_pipeline, self.classification_pipeline = pickle.load(f)
