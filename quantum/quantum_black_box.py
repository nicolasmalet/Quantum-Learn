from dataclasses import dataclass
import numpy as np

from .jpc_chip import JpcChip
from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox
from zeroth.zeroth_order.gradient_estimators import GradientEstimator


@dataclass
class QuantumBlackBoxConfig:
    name: str
    quantum_params: np.ndarray

    def instantiate(self):
        return QuantumBlackBox(self)


class QuantumBlackBox(ZerothOrderBlackBox):
    def __init__(self, config: QuantumBlackBoxConfig):

        self.name: str = config.name
        self.params = config.quantum_params
        self.nb_params = len(self.params)

        self.simulator = JpcChip()

    def get_params(self):
        return self.params

    def init_params(self, quantum_params: np.ndarray) -> None:
        self.params = quantum_params

    def print_params(self):
        print(f"g_conv, g_sq: {self.params}")

    def forward(self, X: np.ndarray):
        """Standard forward pass using the current nominal weights.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Output. Shape: (output_dim, batch_size).
        """
        return self.simulator.run_simulation(X, self.params[None, :])

    def forward_perturbed(self, X: np.ndarray, gradient_estimator: GradientEstimator) -> np.ndarray:
        """Parallel forward pass for multiple perturbed versions of the network.

        This method broadcasts the input X across T perturbed parameter sets
        to compute T outputs simultaneously without a Python loop.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).
            gradient_estimator (GradientEstimator): The gradient_estimator object.

        Returns:
            np.ndarray: Stacked outputs. Shape: (T, output_dim, batch_size)
                        where T is the number of perturbations.
        """
        perturbed_params = gradient_estimator.perturb(self.params)
        return self.simulator.run_simulation(X, perturbed_params)

    def update_params(self, grad: np.ndarray, learning_rate: float) -> None:
        """
        Updates the quantum parameters
        """
        self.params -= learning_rate * grad

