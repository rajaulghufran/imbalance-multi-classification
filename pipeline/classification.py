import re
from datetime import datetime
from time import time

import stanza
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearnex import config_context
from sklearnex.model_selection import train_test_split
from sklearnex.svm import SVC

from .emoticons import EMOTICON_PATTERNS
from .stop_words import STOP_WORDS
from .sub_patterns import SUB_PATTERNS_1, SUB_PATTERNS_2
# from .usm_ndarray_transformer import USMndarrayTransformer

DEFAULT_POS = ["ADJ","ADP","ADV","AUX","CONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
DEFAULT_PARAM_GRID = [
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

def dummy_fun(x):
    return x

class Classification:
    def __init__(self, n_jobs=1, target_offload="auto", verbose=1):
        stanza.download(lang="id")

        self.n_jobs=n_jobs
        self.target_offload=target_offload
        self.verbose=verbose

    def create_pipeline(self):
        tfidfvectorizer_hyperparameters = {
            "strip_accents": "ascii",
            "lowercase": True,
            "preprocessor": dummy_fun,
            "tokenizer": dummy_fun,
            "analyzer": "word",
            "token_pattern": None
        }

        return Pipeline([
            ("tfidfvectorizer", TfidfVectorizer(**tfidfvectorizer_hyperparameters)),
            # ("usm_ndarray_transformer", USMndarrayTransformer()),
            ("svc", SVC(class_weight="balanced", decision_function_shape='ovo', random_state=42)),
        ])

    def clean(self, X):
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: TEXT CLEANING')

        X_cleaned = []

        for x in X:
            string = x

            # normalize unicode to ascii, replace any char to its ascii form or remove it
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

    def tokenize(self, X_cleaned, pos=None):
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: LOAD TOKENIZER', flush=True)

        tokenizer = stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma", use_gpu=True)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: TOKENIZATION')

        X_tokenized = []

        docs = tokenizer.bulk_process(X_cleaned)

        if pos is None:
            pos = DEFAULT_POS

        for doc in docs:
            doc_lemma = []

            for sentence in doc.sentences:
                for token in sentence.tokens:
                    for word in token.words:
                        if word.pos in pos:
                            lemma = word.lemma

                            if lemma not in STOP_WORDS:
                                doc_lemma.append(lemma)

            X_tokenized.append(doc_lemma)

        return X_tokenized

    def train_test_split(self, X_tokenized, y):
        return train_test_split(
            X_tokenized,
            y,
            test_size=.2,
            random_state=42,
            stratify=y
        )

    def tuning(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: HYPER-PARAMETERS TUNING')

        grid_search = GridSearchCV(
            estimator=self.create_pipeline(),
            param_grid=param_grid,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=self.n_jobs,
            cv=StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42),
            verbose=self.verbose
        )

        start = time()

        with config_context(target_offload=self.target_offload):
            grid_search.fit(X_train, y_train)

        estimation = time() - start

        return(grid_search, estimation)

    def train(self, X_train, y_train, hyperparams):
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TRAINING')

        pipeline = self.create_pipeline()
        pipeline.set_params(**hyperparams)

        with config_context(target_offload=self.target_offload):
            return pipeline.fit(X_train, y_train)
    
    def test(self, model: Pipeline, X_test):
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: MODEL TESTINIG')

        return model.predict(X_test)
    
    def score(self, y_test, y_pred):
        return (
            accuracy_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred)
        )
