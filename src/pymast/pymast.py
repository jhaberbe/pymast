"""Main module."""

import numpy as np
import pandas as pd
from typing import Optional, Union

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

class HurdleLogNormal(BaseEstimator):
    """Thank you https://geoffruddock.com/building-a-hurdle-regression-estimator-in-scikit-learn/ for holding my hand through this process."""

    def __init__(self):

        # Its nice to keep the features.
        self._features = []

        # Ridge regression helps when there is a lot of multicollinearity.
        self.linear = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        self.logistic = LogisticRegression(solver='liblinear') # not sure why any one solver is better than another

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        # Useful validation
        X, y = check_X_y(
            X, 
            y, 
            dtype=None,
            accept_sparse=False, # why not?
            accept_large_sparse=False, # why not?
            force_all_finite='allow-nan'
        )

        # Hurdle Component
        self.logistic.fit(X, y > 0)
        # Linear Component
        self.linear.fit(X, y[y > 0])

        # Done
        self.is_fitted_ = True

        if isinstance(X, pd.DataFrame):
            self._features = X.columns.tolist()
        else:
            self._features = [f'feature_{i}' for i in range(X.shape[1])]

        return self

    def predict(self, X):
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """Generates a prediction for the hurdle portion (0/1), and generates a prediction (float) for the linear portion.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Covariates

        Returns:
            np.array: 1D predictions (float)
        """
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        return self.logistic.predict(X) * self.linear.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        """Generates a probability for the hurdle portion [0, 1], and a prediction (float) for the linear portion.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Covariates

        Returns:
            np.array: 1D predictions (float)
        """
        X = check_array(X, accept_sparse=False, accept_large_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        return self.logistic.predict_proba(X)[:, 1] * self.linear.predict(X) # we grab the second column of the probability matrix (the probability of y > 0)