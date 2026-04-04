import numpy as np
from zeroth.abstract import DataCreator
from zeroth.data import Data

from .config.data_config import nb_period_train, nb_period_test
from .task_constants import NB_POINTS_PER_PERIOD


class DataCreatorSinusHard(DataCreator):
    def __init__(self, nb_periods_train: int, nb_periods_test: int):
        self.nb_periods_train: int = nb_periods_train
        self.nb_periods_test: int = nb_periods_test

    def __call__(self) -> Data:
        sinus = np.array([-0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1])
        square = np.array([1, 1, 1, 1, -1, -1, -1, -1])

        X_train_base = np.empty((self.nb_periods_train, NB_POINTS_PER_PERIOD))
        Y_train_base = np.random.binomial(1, 0.5, self.nb_periods_train)

        for i in range(self.nb_periods_train):
            X_train_base[i, :] = sinus if Y_train_base[i] == 1 else square

        X_train = X_train_base.reshape(-1, 1)
        Y_train = np.repeat(Y_train_base, NB_POINTS_PER_PERIOD).reshape(-1, 1)

        X_test_base = np.empty((self.nb_periods_test, NB_POINTS_PER_PERIOD))
        Y_test_base = np.random.binomial(1, 0.5, (self.nb_periods_test, 1))

        for i in range(self.nb_periods_test):
            X_test_base[i, :] = sinus if Y_test_base[i] == 1 else square

        X_test = X_test_base.reshape(-1, 1)
        Y_test = np.repeat(Y_test_base, NB_POINTS_PER_PERIOD).reshape(-1, 1)

        return Data(raw_X_train=X_train, raw_Y_train=Y_train, raw_X_test=X_test, raw_Y_test=Y_test, nb_class=2)


data_creator = DataCreatorSinusHard(nb_periods_train=nb_period_train,
                                    nb_periods_test=nb_period_test)
