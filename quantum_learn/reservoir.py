import numpy as np
from jax import Array

from .types import Array


class Reservoir:
    def __init__(self) -> None:
        self.W: Array = np.array([])

    def __call__(self, F: Array) -> Array:
        return self.forward(F)

    def fit(self, F: Array, Y: Array) -> None:
        F_inv = np.linalg.pinv(F)
        self.W = F_inv @ Y

    def forward(self, F: Array) -> Array:
        Y = F @ self.W
        return Y
