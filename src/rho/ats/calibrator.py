"""Maps rho's raw ComponentVector onto real ATS-engine outcome (0..100)."""

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from rho.models.scoring import ComponentVector

# Frozen feature order — a saved calibrator is only valid for this ordering.
FEATURES = [
    "keyword_coverage",
    "semantic_similarity",
    "fuzzy_coverage",
    "must_have_coverage",
    "nice_have_coverage",
]


def _vec(cv: ComponentVector) -> list[float]:
    return [getattr(cv, f) for f in FEATURES]


class Calibrator:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, X: list[ComponentVector], y: list[float]) -> None:
        self.model.fit(np.array([_vec(x) for x in X]), np.array(y))
        self._fitted = True

    def predict(self, cv: ComponentVector) -> float:
        if not self._fitted:
            raise RuntimeError("calibrator not fitted")
        p = float(self.model.predict(np.array([_vec(cv)]))[0])
        return max(0.0, min(100.0, p))

    def save(self, path) -> None:
        joblib.dump(self.model, path)

    def load(self, path) -> "Calibrator":
        self.model = joblib.load(path)
        self._fitted = True
        return self
