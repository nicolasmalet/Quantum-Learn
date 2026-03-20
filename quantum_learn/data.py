import numpy as np

from lab.sinus_vs_square_hard.task_constants import NB_POINTS_PER_PERIOD
from quantum_learn.types import Array


class DataSignal:
    def __init__(self, raw_X_train: Array, raw_Y_train: Array, raw_X_test: Array,
                 raw_Y_test: Array):
        self.input_dim: int = 1
        self.output_dim: int = raw_Y_train.shape[0]
        self.nb_data: int = raw_X_train.shape[0]
        self.nb_periods = self.nb_data // NB_POINTS_PER_PERIOD
        self.nb_tests: int = raw_X_test.shape[0]

        self.nb_periods_per_batch: int | None = None
        self.batch_size: int | None = None
        self.nb_batches: int | None = None

        self.raw_X_train: Array = raw_X_train
        self.raw_Y_train: Array = raw_Y_train
        self.X_test: Array = raw_X_test
        self.Y_test: Array = raw_Y_test

        self.X_train: Array = np.array([])
        self.Y_train: Array = np.array([])

    def prepare_data(self, nb_periods_per_batch: int) -> None:
        self.nb_periods_per_batch = nb_periods_per_batch
        self.batch_size = nb_periods_per_batch
        self.nb_batches = self.nb_data // self.batch_size

        self.X_train = np.reshape(self.raw_X_train, (-1, NB_POINTS_PER_PERIOD * self.batch_size))
        self.Y_train = np.reshape(self.raw_Y_train, (-1, 1, NB_POINTS_PER_PERIOD * self.batch_size))


def create_data(nb_periods_train: int = 100, nb_periods_test: int = 10) -> DataSignal:
    sinus = np.array([-0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1])
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    X_train = np.zeros([NB_POINTS_PER_PERIOD * nb_periods_train])
    Y_train = np.random.binomial(1, 0.5, [nb_periods_train])

    for i in range(nb_periods_train):
        X_train[NB_POINTS_PER_PERIOD * i: NB_POINTS_PER_PERIOD * (i + 1)] = sinus if Y_train[i] == 1 else square

    Y_train = Y_train.repeat(NB_POINTS_PER_PERIOD)[None, :]

    X_test = np.zeros([NB_POINTS_PER_PERIOD * nb_periods_test])
    Y_test = np.random.binomial(1, 0.5, [nb_periods_test])

    for i in range(nb_periods_test):
        X_test[NB_POINTS_PER_PERIOD * i: NB_POINTS_PER_PERIOD * (i + 1)] = sinus if Y_test[i] == 1 else square

    Y_test = Y_train.repeat(NB_POINTS_PER_PERIOD)[None, :]

    return DataSignal(X_train, Y_train, X_test, Y_test)
