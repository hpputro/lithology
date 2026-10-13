import warnings
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from litho_setup import *

warnings.filterwarnings("ignore", category=FutureWarning)


class MeanImputer:
    def fit(self, X):
        self.mu_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        out = X.copy()
        r, c = np.where(np.isnan(out))
        out[r, c] = self.mu_[c]
        return out


def _lgbm(seed):
    return LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": LGBM_N_EST}, random_state=seed)


class LGBMImputer:
    """One regressor per target, single pass, NaN handled natively.
    use_targets=False -> predictors-only (no information from other logs).
    use_targets=True  -> direct (features = predictors + the 2 other targets)."""
    def __init__(self, seed=0, use_targets=True):
        self.seed, self.use_targets = seed, use_targets

    def _feats(self, j, n_cols):
        if not self.use_targets:
            return list(range(N_PRED))
        return [c for c in range(n_cols) if c != j]

    def fit(self, X):
        self.models_ = {}
        for j in range(N_PRED, X.shape[1]):
            feats = self._feats(j, X.shape[1])
            obs = ~np.isnan(X[:, j])
            m = _lgbm(self.seed).fit(X[obs][:, feats], X[obs, j])
            self.models_[j] = (m, feats)
        return self

    def transform(self, X):
        out = X.copy()
        for j, (m, feats) in self.models_.items():
            miss = np.isnan(X[:, j])
            if miss.any():
                out[miss, j] = m.predict(X[miss][:, feats])  # features from X, not from out
        return out


class LGBMChainedImputer:
    """Chained/iterative: IterativeImputer with LGBM as estimator. Imputed values are fed back
    as features in the next round. Clipped to the training range (as before)."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.imp = IterativeImputer(estimator=_lgbm(self.seed), max_iter=CHAIN_MAX_ITER,
                                    random_state=self.seed, initial_strategy="mean",
                                    min_value=np.nanmin(X, axis=0), max_value=np.nanmax(X, axis=0))
        self.imp.fit(X)
        return self

    def transform(self, X):
        return self.imp.transform(X)


def make_imputers(seed):
    return {"Mean": MeanImputer(),
            "LGBM-PredOnly": LGBMImputer(seed, use_targets=False),
            "LGBM-Direct": LGBMImputer(seed, use_targets=True),
            "LGBM-Chained": LGBMChainedImputer(seed)}
