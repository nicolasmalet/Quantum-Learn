import numpy as np
from zeroth.abstract import DataCreator
from zeroth.data import Data

from .task_constants import NB_POINTS_PER_PERIOD, sinus, square


class DataCreatorSinusHard(DataCreator):
    def __init__(self, nb_periods_train: int, nb_periods_test: int):
        self.nb_periods_train: int = nb_periods_train
        self.nb_periods_test: int = nb_periods_test

    def __call__(self) -> Data:
        X_train, Y_train = create_data(self.nb_periods_train)
        X_test, Y_test = create_data(self.nb_periods_test)

        return Data(raw_X_train=X_train, raw_Y_train=Y_train, raw_X_test=X_test, raw_Y_test=Y_test)


def create_data(nb_periods: int) -> tuple[np.ndarray, np.ndarray]:
    X_base = np.empty((nb_periods, NB_POINTS_PER_PERIOD))
    Y_base = np.random.binomial(1, 0.5, nb_periods)

    for i in range(nb_periods):
        X_base[i, :] = sinus if Y_base[i] == 1 else square

    X = X_base.reshape(-1, 1)
    Y = np.repeat(Y_base, NB_POINTS_PER_PERIOD).reshape(-1, 1)

    return X, Y


data_creator = DataCreatorSinusHard(nb_periods_train=300,
                                    nb_periods_test=10)
