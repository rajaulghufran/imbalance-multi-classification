from datetime import datetime
from typing import List, Set, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document, Sentence, Token, Word

from .data.stopwords import STOPWORDS

class StopWordRemoval(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        stopwords: Union[int, None, Union[List[str], Set[str], Tuple[str]]] = -1,
        verbose: int = 1
    ) -> None:
        if stopwords == -1:
            self.stopwords = [x.lower() for x in STOPWORDS]
        else:
            self.stopwords = stopwords

        self.verbose = verbose

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Document], y=None) -> List[Document]:
        if self.verbose > 1:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: STOPWORD REMOVAL')

        X_documents = X.copy()

        if self.stopwords is not None:

            for di, document in enumerate(X_documents):
                sentences: List[Sentence] = document.sentences

                for si, sentence in enumerate(sentences):
                    tokens: List[Token] = sentence.tokens

                    for ti, token in enumerate(tokens):
                        words: List[Word] = token.words

                        for wi, word in enumerate(words):
                            word_dict = word.to_dict()

                            if any(
                                word_dict.get(attr) in self.stopwords
                                for attr in ["lemma", "text"]
                            ):
                                del X_documents[di].sentences[si].tokens[ti].words[wi]

        return X_documents
