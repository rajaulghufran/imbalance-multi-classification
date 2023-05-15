import re
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
from tqdm import tqdm

from .assets.emoticons import EMOTICON_PATTERNS
from .assets.stop_words import STOP_WORDS
from .assets.sub_patterns import SUB_PATTERNS_1, SUB_PATTERNS_2
# from .usm_ndarray_transformer import USMndarrayTransformer

POS = {
    "tags": ["ADJ","ADP","ADV","AUX","CONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"],
    "descriptions": ["adjective","adposition","adverb","auxiliary","conjunction","determiner","interjection","noun","numeral","particle","pronoun","proper noun","punctuation","subordinating conjunction","symbol","verb","other"],
    "examples": ["salah, pertama, besar","di, pada, dari","juga, lebih, kemudian","adalah, akan, dapat","dan, atau, tetapi","ini, itu, buah","Hai, Oh, Sayang","tahun, orang, desa","satu, dua, 1","tidak, kah, lah","yang, dia, mereka","Indonesia, kabupaten, kecamatan",", ? ()","untuk, bahwa, dengan","%, =, °","menjadi, merupakan, memiliki", "and, image, in"]
}

def dummy_fun(x):
    return x

class Classification:
    def __init__(self, n_jobs=1, target_offload="auto"):
        stanza.download(lang="id")

        self.n_jobs=n_jobs
        self.target_offload=target_offload

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

    def clean(self, X):
        X_cleaned = []

        for x in tqdm(X, desc="Text cleaning"):
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

    def tokenize(self, X_cleaned, selected_pos=None):
        tokenizer = stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma", use_gpu=True)

        X_tokenized = []

        docs = tokenizer.bulk_process(X_cleaned)

        if selected_pos is None:
            selected_pos = POS["tags"].copy()

        for doc in tqdm(docs, desc="Lemmatization"):
            doc_lemma = []

            for sentence in doc.sentences:
                for token in sentence.tokens:
                    for word in token.words:
                        if word.pos in selected_pos:
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
            n_jobs=self.n_jobs,
            cv=StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=42),
            verbose=3
        )

        start = time()

        with config_context(target_offload=self.target_offload):
            grid_search.fit(X_train, y_train)

        estimation = time() - start

        return(grid_search.best_params_, estimation, grid_search)

    def train(self, X_train, y_train, hyperparams):
        pipeline = self.create_pipeline()
        pipeline.set_params(**hyperparams)

        with config_context(target_offload=self.target_offload):
            return pipeline.fit(X_train, y_train)
    
    def test(self, model: Pipeline, X_test):
        return model.predict(X_test)
    
    def score(self, y_test, y_pred):
        return (
            accuracy_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred)
        )
