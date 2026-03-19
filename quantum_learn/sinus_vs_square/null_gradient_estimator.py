from zeroth.zeroth_order import GradientEstimator, GradientEstimatorConfig

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NullGradientEstimatorConfig(GradientEstimatorConfig):
    def instantiate(self, nb_params: int):
        return NullGradientEstimator(nb_params=nb_params, dA=self.dA)



class NullGradientEstimator(GradientEstimator):
    def __init__(self, nb_params, dA):
        super().__init__(nb_params, dA)
        self.nb_params: int = nb_params
        self.Ps: np.ndarray = np.ndarray([])

    def perturb(self, Theta: np.ndarray):
        return Theta[None, :]

    def get_gradient(self, p_Loss: np.ndarray) -> np.ndarray:
        return np.zeros(self.nb_params)