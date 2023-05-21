from datetime import datetime
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document, Sentence, Token, Word

POS = set(["ADJ","ADP","ADV","AUX","CONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"])

class POSFilter(BaseEstimator, TransformerMixin):
    def __init__(self, pos=None) -> None:
        self.pos = pos

        if self.pos is None:
            self.pos = POS

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Document], y=None) -> List[Document]:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: POS REMOVAL')

        X_documents = X.copy()

        for di, document in enumerate(X):
            sentences: List[Sentence] = document.sentences

            for si, sentence in enumerate(sentences):
                tokens: List[Token] = sentence.tokens

                for ti, token in enumerate(tokens):
                    words: List[Word] = token.words

                    for wi, word in enumerate(words):
                        if word.pos not in self.pos:
                            del X_documents[di].sentences[si].tokens[ti].words[wi]

        return X_documents