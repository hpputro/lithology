import warnings
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from litho_setup import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")


class MeanImputer:
    def fit(self, X):
        self.mu_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        out = X.copy()
        r, c = np.where(np.isnan(out))
        out[r, c] = self.mu_[c]
        return out


class MICEImputer:
    """IterativeImputer; imputed values clipped to the observed training range per column
    (prevents divergence, which 10**RS would otherwise amplify)."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.imp = IterativeImputer(max_iter=MICE_MAX_ITER, random_state=self.seed, sample_posterior=False,
                                    min_value=np.nanmin(X, axis=0), max_value=np.nanmax(X, axis=0))
        self.imp.fit(X)
        return self

    def transform(self, X):
        return self.imp.transform(X)


class LGBMImputer:
    """One regressor per target; features = all other columns (NaN handled natively)."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.models_ = {}
        for j in range(N_PRED, X.shape[1]):
            feats = [c for c in range(X.shape[1]) if c != j]
            obs = ~np.isnan(X[:, j])
            Xtr, Xva, ytr, yva = train_test_split(X[obs][:, feats], X[obs, j],
                                                  test_size=VAL_SIZE, random_state=self.seed)
            m = LGBMRegressor(**LGBM_PARAMS, random_state=self.seed)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            self.models_[j] = (m, feats)
        return self

    def transform(self, X):
        out = X.copy()
        for j, (m, feats) in self.models_.items():
            miss = np.isnan(X[:, j])
            if miss.any():
                out[miss, j] = m.predict(X[miss][:, feats])  # single pass, features from X
        return out


def make_imputers(seed):
    return {"Mean": MeanImputer(), "MICE": MICEImputer(seed), "LGBM": LGBMImputer(seed)}
