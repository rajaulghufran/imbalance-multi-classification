from datetime import datetime
from typing import List

import stanza
from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document

class TokenizeMWTPOSLemma(BaseEstimator, TransformerMixin):
    def __init__(self, verbose: int = 1) -> None:
        self.verbose = verbose

        if self.verbose > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: DOWNLOAD STANZA MODEL')

        if self.verbose > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: LOAD STANZA PIPELINE: tokenize,mwt,pos,lemma')

        self.tokenizer = stanza.Pipeline("id", processors="tokenize,mwt,pos,lemma", use_gpu=True)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[str], y=None) -> List[Document]:
        if self.verbose > 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: TOKENIZE, MWT, POS, LEMMA')

        return self.tokenizer.bulk_process(X)
