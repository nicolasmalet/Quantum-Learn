import numpy as np


class DataSignal:
    def __init__(self, raw_X_train: np.ndarray, raw_Y_train: np.ndarray, raw_X_test: np.ndarray, raw_Y_test: np.ndarray):
        self.input_dim: int = raw_X_train.shape[0]
        self.output_dim: int = raw_Y_train.shape[0]
        self.nb_data: int = raw_X_train.shape[1]
        self.nb_tests: int = raw_X_test.shape[1]

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

        self.X_train = np.reshape(self.raw_X_train, (-1, self.input_dim, self.batch_size))
        self.Y_train = np.reshape(self.raw_Y_train, (-1, self.output_dim, self.batch_size))


def create_data_circle(nb_points_train: int, nb_points_test, r: float) -> DataSignal:

    X_train = np.random.uniform(0, 1, (2, nb_points_train))
    Y_train = (X_train[0, :] ** 2 + X_train[1, :] ** 2 < r ** 2).astype(np.int8)[None, :]

    X_test = np.random.uniform(0, 1, (2, nb_points_test))
    Y_test = (X_test[0, :] ** 2 + X_test[1, :] ** 2 < r ** 2).astype(np.int8)[None, :]

    data_sinus_vs_square = DataSignal(X_train, Y_train, X_test, Y_test)

    return data_sinus_vs_square


data = create_data_circle(10, 10, 0.7)
data.prepare_data(5)
