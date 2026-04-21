import numpy as np

from .types import Array
from zeroth.abstract import BlackBox

class Reservoir(BlackBox):
    def __init__(self) -> None:
        self.W: Array | float = 0

    def __call__(self, F: Array) -> Array:
        return self.forward(F)

    def fit(self, F: Array, Y: Array) -> None:
        self.W = self.compute_W(F, Y)

    def forward(self, F: Array) -> Array:
        Y = F @ self.W
        return Y

    @staticmethod
    def compute_W(F: Array, Y: Array) -> Array:
        # F_inv = np.linalg.pinv(F)
        # W = F_inv @ Y
        W, _, _, _ = np.linalg.lstsq(F, Y)
        return W

    def get_params(self) -> dict:
        return {"W": self.W}

    def init_params(self, params: dict) -> None:
        self.W = params["W"]
