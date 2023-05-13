from sklearn.base import BaseEstimator, TransformerMixin

import dpctl.tensor as dpt

class USMndarrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return dpt.from_numpy(X.toarray())
