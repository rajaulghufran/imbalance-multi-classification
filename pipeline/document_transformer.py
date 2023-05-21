from datetime import datetime
from typing import List, Union

from sklearn.base import BaseEstimator, TransformerMixin
from stanza.models.common.doc import Document, Sentence, Token, Word

class DocumentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_attrs: Union[None, List[str]] = None):
        # reduce stanza document to a sequence of word properties joined by dot (.)
        # see https://stanfordnlp.github.io/stanza/data_objects.html on word properties

        self.feat_attrs = feat_attrs

        if self.feat_attrs is None:
            self.feat_attrs = ["lemma"]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Document], y=None, verbose__: bool = True) -> List[List[str]]:
        if verbose__:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} INFO: DOCUMENT TRANSFORMER')

        X_documents: List[List[str]] = []

        for document in X:
            new_document = []

            sentences: List[Sentence] = document.sentences

            for sentence in sentences:
                tokens: List[Token] = sentence.tokens

                for token in tokens:
                    words: List[Word] = token.words

                    for word in words:
                        new_document.append(
                            ".".join([
                                str(
                                    word
                                    .to_dict()
                                    .get(key, None)
                                )
                                for key in self.feat_attrs
                            ])
                        )

            X_documents.append(new_document)

        return X_documents