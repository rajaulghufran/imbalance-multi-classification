import re
from time import time

import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearnex import config_context
from sklearnex.model_selection import train_test_split
from sklearnex.svm import SVC
from tqdm import tqdm

from .stop_words import STOP_WORDS
# from .usm_ndarray_transformer import USMndarrayTransformer

def dummy_fun(x):
    return x

class Classification:
    def __init__(self):
        stanza.download("id")

        self.tokenizer = stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma", use_gpu=True)

        self.param_grid = {
            "tfidfvectorizer__norm": ("l1", "l2"),
            "tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (1,3)),
            "tfidfvectorizer__min_df": (1, 3, 5, 10),
            "tfidfvectorizer__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
            "svc__kernel": ("linear", "rbf"),
            "svc__C": (0.01, 0.1, 1, 10, 100, 1000, 10000),
            "svc__gamma": (0.0001, 0.001, 0.01, 0.1, 1),
            "svc__decision_function_shape": ("ovo", "ovr")
        }

    def create_pipeline(self):
        tfidfvectorizer_hyperparameters = {
            "strip_accents": "ascii",
            "lowercase": True,
            "preprocessor": dummy_fun,
            "tokenizer": dummy_fun,
            "analyzer": "word",
            "stop_words": sorted(list(STOP_WORDS)),
            "token_pattern": None
        }

        return Pipeline([
            ("tfidfvectorizer", TfidfVectorizer(**tfidfvectorizer_hyperparameters)),
            # ("usm_ndarray_transformer", USMndarrayTransformer()),
            ("svc", SVC(class_weight="balanced", random_state=42)),
        ])

    def get_params(self, pipeline_name, val_to_str = False):
        params = {}

        for key, val in self.param_grid.items():
            if pipeline_name in key:
                params[key.split(pipeline_name + "__")[-1]] = [str(x) if val_to_str == True else x for x in val]

        return params

    def set_param_grid_attr(self, key, val):
        self.param_grid.update({key: val})

    #TODO:
    def clean(self, X):
        X_cleaned = []

        for x in tqdm(X, desc="Text cleaning"):
            text = x
            text = re.sub("\s{2,}", " ", text)

            X_cleaned.append(text)

        return X_cleaned

    def tokenize(self, X_cleaned):
        X_tokenized = []

        docs = self.tokenizer.bulk_process(X_cleaned)

        for doc in tqdm(docs, desc="Lemmatization"):
            doc_lemma = []

            for sentence in doc.sentences:
                for token in sentence.tokens:
                    for word in token.words:
                        doc_lemma.append(word.lemma)

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

    def tuning(self, X_train, y_train):
        grid_search = GridSearchCV(
            estimator=self.create_pipeline(),
            param_grid=self.param_grid,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=-1,
            cv=StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42),
            verbose=3
        )

        start = time()

        with config_context(target_offload="auto"):
            grid_search.fit(X_train, y_train)

        estimation = time() - start

        return(grid_search.best_params_, estimation, grid_search)

    def train(self, X_train, y_train, hyperparams):
        pipeline = self.create_pipeline()
        pipeline.set_params(**hyperparams)

        with config_context(target_offload="auto"):
            return pipeline.fit(X_train, y_train)
    
    def test(self, model: Pipeline, X_test):
        return model.predict(X_test)
    
    def score(self, y_test, y_pred):
        return (
            confusion_matrix(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred)
        )
