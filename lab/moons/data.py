import numpy as np
from sklearn.datasets import make_moons
from zeroth.abstract import DataCreator
from zeroth.data import Data


class DataCreatorMoons(DataCreator):
    def __init__(self, n_samples_train: int, n_samples_test: int, noise: float = 0.1):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.noise = noise

    def __call__(self) -> Data:
        X_train, Y_train = create_moon_data(self.n_samples_train, self.noise)
        X_test, Y_test = create_moon_data(self.n_samples_test, self.noise)

        return Data(
            raw_X_train=X_train,
            raw_Y_train=Y_train,
            raw_X_test=X_test,
            raw_Y_test=Y_test,
            nb_class=2
        )


def create_moon_data(n_samples: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    X_pts, Y_pts = make_moons(n_samples=n_samples, noise=noise)
    X = X_pts.reshape(-1, 1)
    Y = np.repeat(Y_pts, 2).reshape(-1, 1)

    return X, Y


# Instanciation par défaut avec 300 échantillons (correspondant à vos 300 périodes habituelles)
data_creator = DataCreatorMoons(
    n_samples_train=300,
    n_samples_test=100,
    noise=0.1
)