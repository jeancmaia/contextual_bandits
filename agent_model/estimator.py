import numpy as np

from joblib import wrap_non_picklable_objects
from copy import deepcopy
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import AdaptiveGreedy


@wrap_non_picklable_objects
class Agent:
    def __init__(self, n_choices=4, batch_size=15):
        self.__nchoices = n_choices
        self.__batch_size = batch_size

        ols = LinearRegression(lambda_=1.0, fit_intercept=True, method="sm")

        beta_prior = ((3.0 / self.__nchoices, 4), 2)

        self._model = AdaptiveGreedy(
            deepcopy(ols),
            nchoices=self.__nchoices,
            smoothing=(1, 2),
            beta_prior=beta_prior,
            batch_train=True,
            random_state=12,
        )

    def fit(self, covariates, action_chosen, rewards_received):
        self._model.fit(X=covariates, a=action_chosen, r=rewards_received)

    def partial_fit(self, covariates, action_chosen, rewards_received):
        self._model.partial_fit(covariates, action_chosen, rewards_received)

    def predict(self, X):
        prediction = self._model.predict(X)
        if prediction is None:
            return np.random.randint(self.__nchoices, size=self.__batch_size)
        else:
            return prediction.astype("uint8")
