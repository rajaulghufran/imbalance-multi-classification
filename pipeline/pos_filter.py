from datetime import datetime
from typing import List, Set, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document, Sentence, Token, Word

POS = set(["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"])

class POSFilter(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            pos: Union[int, None, Union[List[str], Set[str], Tuple[str]]] = -1,
            verbose: int = 1
        ) -> None:
        if self.pos == -1:
            self.pos = POS
        else:
            self.pos = pos

        self.verbose = verbose

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Document], y=None) -> List[Document]:
        if self.verbose > 0:
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
