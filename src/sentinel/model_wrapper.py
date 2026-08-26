"""LightGBM wrapper for sklearn compatibility (needed by calibrator)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster=None):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        return np.column_stack([1 - raw, raw])

    def predict(self, X):
        return (self.booster.predict(X) > 0.5).astype(int)