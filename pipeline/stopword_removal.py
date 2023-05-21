from datetime import datetime
from typing import List, Literal, Set, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document, Sentence, Token, Word

from .data.stopwords import STOPWORDS

class StopWordRemoval(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        stopwords: Union[None, Union[List[str], Set[str], Tuple[str]]] = None,
        word_attr: Literal["text", "lemma"] = "lemma"
    ) -> None:
        self.word_attr = word_attr

        if stopwords is None:
            stopwords = STOPWORDS

        self.stopwords = [x.lower() for x in stopwords]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Document], y=None) -> List[Document]:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: STOPWORD REMOVAL')

        X_documents = X.copy()

        for di, document in enumerate(X_documents):
            sentences: List[Sentence] = document.sentences

            for si, sentence in enumerate(sentences):
                tokens: List[Token] = sentence.tokens

                for ti, token in enumerate(tokens):
                    words: List[Word] = token.words

                    for wi, word in enumerate(words):
                        lemma: str = word.lemma

                        if lemma.lower() in self.stopwords:
                            del X_documents[di].sentences[si].tokens[ti].words[wi]

        return X_documents
