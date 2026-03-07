import numpy as np


class DataSignal:
    def __init__(self, nb_points_per_period, raw_X_train: np.ndarray, raw_Y_train: np.ndarray, raw_X_test: np.ndarray, raw_Y_test: np.ndarray):
        self.nb_points_per_period = nb_points_per_period
        self.input_dim: int = 1
        self.output_dim: int = raw_Y_train.shape[0]
        self.nb_data: int = raw_X_train.shape[0]
        self.nb_periods = self.nb_data // self.nb_points_per_period
        self.nb_tests: int = raw_X_test.shape[0]

        self.batch_size: int | None = None
        self.nb_batches: int | None = None

        self.raw_X_train: np.ndarray = raw_X_train
        self.raw_Y_train: np.ndarray = raw_Y_train
        self.X_test: np.ndarray = raw_X_test
        self.Y_test: np.ndarray = raw_Y_test

        self.X_train: np.ndarray = np.array([])
        self.Y_train: np.ndarray = np.array([])

    def prepare_data(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.nb_batches = self.nb_data // self.batch_size

        self.X_train = np.reshape(self.raw_X_train, (-1, self.nb_points_per_period * self.batch_size))
        self.Y_train = np.reshape(self.raw_Y_train, (-1, 1, self.batch_size))


def create_data_sinus_vs_square(nb_periods_train, nb_periods_test) -> DataSignal:
    nb_points_per_period = 8
    sinus = np.array([-0.7, 0, 0.7, 1, 0.7, 0, -0.7, -1])
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    X_train = np.zeros([nb_points_per_period * nb_periods_train])
    Y_train = np.random.binomial(1, 0.5, [1, nb_periods_train])

    for i in range(nb_periods_train):
        X_train[nb_points_per_period*i: nb_points_per_period * (i + 1)] = sinus if Y_train[0][i] == 1 else square

    X_test = np.zeros([nb_points_per_period * nb_periods_test])
    Y_test = np.random.binomial(1, 0.5, [1, nb_periods_test])

    for i in range(nb_periods_test):
        X_test[nb_points_per_period*i: nb_points_per_period * (i + 1)] = sinus if Y_test[0][i] == 1 else square

    data_sinus_vs_square = DataSignal(nb_points_per_period, X_train, Y_train, X_test, Y_test)

    return data_sinus_vs_square
