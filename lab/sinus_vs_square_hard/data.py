import numpy as np
from zeroth.data.data import Data
from .task_constants import NB_POINTS_PER_PERIOD


def create_data(nb_periods_train: int = 300, nb_periods_test: int = 10) -> Data:
    sinus = np.array([-0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1])
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    X_train_base = np.empty((nb_periods_train, NB_POINTS_PER_PERIOD))
    Y_train_base = np.random.binomial(1, 0.5, nb_periods_train).astype(np.float64)

    for i in range(nb_periods_train):
        X_train_base[i, :] = sinus if Y_train_base[i] == 1 else square

    X_train = X_train_base.reshape(-1, 1)
    Y_train = np.repeat(Y_train_base, NB_POINTS_PER_PERIOD).reshape(-1, 1)

    X_test_base = np.empty((nb_periods_test, NB_POINTS_PER_PERIOD))
    Y_test_base = np.random.binomial(1, 0.5, (nb_periods_test, 1)).astype(np.float64)

    for i in range(nb_periods_test):
        X_test_base[i, :] = sinus if Y_test_base[i] == 1 else square

    X_test = X_test_base.reshape(-1, 1)
    Y_test = np.repeat(Y_test_base, NB_POINTS_PER_PERIOD).reshape(-1, 1)

    return Data(X_train, Y_train, X_test, Y_test)
