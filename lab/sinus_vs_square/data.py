from functools import partial

import numpy as np
from zeroth.data.data import Data

from .config.data_config import nb_period_train, nb_period_test
from .task_constants import NB_POINTS_PER_PERIOD


def create_data(nb_periods_train: int, nb_periods_test: int) -> Data:
    sinus = np.array([-0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1])
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    X_train = np.empty((nb_periods_train, NB_POINTS_PER_PERIOD))
    Y_train = np.random.binomial(1, 0.5, (nb_periods_train, 1))

    for i in range(nb_periods_train):
        X_train[i, :] = sinus if Y_train[i, 0] == 1 else square

    X_test = np.empty((nb_periods_test, NB_POINTS_PER_PERIOD))
    Y_test = np.random.binomial(1, 0.5, (nb_periods_test, 1))

    for i in range(nb_periods_test):
        X_test[i, :] = sinus if Y_test[i, 0] == 1 else square

    return Data(X_train, Y_train, X_test, Y_test)


create_data_default = partial(
    create_data,
    nb_periods_train=nb_period_train,
    nb_periods_test=nb_period_test
)
