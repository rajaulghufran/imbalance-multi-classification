import pickle
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Literal, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.svm import LinearSVC

from .document_transformer import DocumentTransformer
from .pos_filter import POSFilter
from .stopword_removal import StopWordRemoval
from .text_cleaning import TextCleaning
from .tokenize_mwt_pos_lemma import TokenizeMWTPOSLemma

def fun(arg):
    return arg

class Classification:
    def __init__(self) -> None:
        self.text_preprocessing_pipeline: Pipeline = Pipeline([
            ("text_cleaning", TextCleaning()),
            ("tokenize_mwt_pos_lemma", TokenizeMWTPOSLemma())
        ])

        self.feature_selection_pipeline: Pipeline = Pipeline([
            ("pos_filter", POSFilter()),
            ("stopword_removal", StopWordRemoval()),
            ("document_transformer", DocumentTransformer())
        ])

        tfidfvectorizer_hyperparameters = {
            "encoding": "ascii",
            "decode_error": "ignore",
            "strip_accents": "ascii",
            "preprocessor": fun,
            "tokenizer": fun,
            "analyzer": "word",
            "token_pattern": None,
        }

        self.classification_pipeline: Pipeline = Pipeline([
            ("tfidfvectorizer", TfidfVectorizer(**tfidfvectorizer_hyperparameters)),
            ("linearsvc", LinearSVC(random_state=42)),
        ])

    def get_model_attrs(self):
        pos_filter = self.feature_selection_pipeline.named_steps["pos_filter"]
        stopword_removal = self.feature_selection_pipeline.named_steps["stopword_removal"]
        document_transformer = self.feature_selection_pipeline.named_steps["document_transformer"]
        tfidfvectorizer = self.classification_pipeline.named_steps["tfidfvectorizer"]
        linearsvc = self.classification_pipeline.named_steps["linearsvc"]

        attrs = {
            "pos_filter__pos": pos_filter.pos,
            "stopword_removal__stopwords": set(stopword_removal.stopwords),
            "document_transformer__feat_attrs": document_transformer.feat_attrs,
            "tfidfvectorizer__ngram_range": tfidfvectorizer.ngram_range,
            "tfidfvectorizer__min_df": tfidfvectorizer.min_df,
            "tfidfvectorizer__max_df": tfidfvectorizer.max_df,
            "tfidfvectorizer__norm": tfidfvectorizer.norm,
            "tfidfvectorizer__sublinear_tf": tfidfvectorizer.sublinear_tf,
            "tfidfvectorizer__stop_words": tfidfvectorizer.stop_words_,
            "tfidfvectorizer__vocabulary": tfidfvectorizer.vocabulary_,
            "linearsvc__penalty": linearsvc.penalty,
            "linearsvc__loss": linearsvc.loss,
            "linearsvc__dual": linearsvc.dual,
            "linearsvc__tol": linearsvc.tol,
            "linearsvc__C": linearsvc.C,
            "linearsvc__multi_class": linearsvc.multi_class,
            "linearsvc__fit_intercept": linearsvc.fit_intercept,
            "linearsvc__intercept_scaling": linearsvc.intercept_scaling,
            "linearsvc__class_weight": linearsvc.class_weight,
            "linearsvc__max_iter": linearsvc.max_iter,
            "linearsvc__classes": linearsvc.classes_
        }

        return attrs

    def tuning(
        self,
        X_train: Iterable,
        y_train: Iterable,
        param_distributions: Union[None, List[Dict[str, any]]] = None,
        n_iter: int = 10,
        n_splits: int = 5,
        train_size: Union[float, int] = 0.8,
        n_jobs: int = 1,
        verbose: Literal[1, 2, 3] = 3
    ) -> Tuple[RandomizedSearchCV, float]:
        if param_distributions is None:
            param_distributions = {
                "tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)),
                "tfidfvectorizer__min_df": (1, 3, 5, 10, 25),
                "tfidfvectorizer__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
                "tfidfvectorizer__norm": (None, "l1", "l2"),
                "tfidfvectorizer__sublinear_tf": (True, False),
                "linearsvc__penalty": ("l1","l2"),
                "linearsvc__loss": ("squared_hinge",),
                "linearsvc__dual": (False,),
                "linearsvc__tol": (0.0001,),
                "linearsvc__C": (0.001, 0.01, 0.1, 1, 10, 100, 1000),
                "linearsvc__multi_class": ("ovr",),
                "linearsvc__fit_intercept": (True, False),
                "linearsvc__intercept_scaling": (0.001, 0.01, 0.1, 1.0, 10, 100, 1000),
                "linearsvc__class_weight": (None, "balanced"),
                "linearsvc__max_iter": (100000,)
            }
            
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: HYPER-PARAMETERS TUNING')

        randomized_search = RandomizedSearchCV(
            estimator=self.classification_pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=n_jobs,
            cv=StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=42),
            verbose=verbose,
            random_state=42
        )

        t0 = time()

        randomized_search.fit(X_train, y_train)

        return (randomized_search, time() - t0)

    def train(self, X_train: Iterable, y_train: Iterable) -> None:
        X_train = self.text_preprocessing_pipeline.transform(X_train)
        X_train = self.feature_selection_pipeline.transform(X_train)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TRAINING')
        self.classification_pipeline.fit(X_train, y_train)

    def train_preprocessed(self, X_train: Iterable, y_train: Iterable) -> None:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TRAINING')
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
            check_is_fitted(self.classification_pipeline.named_steps["linearsvc"], ["classes_"])
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
