from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox
from zeroth.zeroth_order.gradient_estimators import GradientEstimator

from quantum_simulation.simulation_params import SimulationParams
from quantum_simulation.quantum_params import QuantumParams
from quantum_simulation.quadrature import Quadrature
from quantum_simulation.jpc_chip import JpcChip

from dataclasses import dataclass
import numpy as np


@dataclass
class QuantumBlackBoxConfig:
    name: str
    quantum_params: np.ndarray
    quantum_parameters: QuantumParams
    simulation_params: SimulationParams

    def instantiate(self):
        return QuantumBlackBox(self)


class QuantumBlackBox(ZerothOrderBlackBox):
    def __init__(self, config: QuantumBlackBoxConfig):

        self.name: str = config.name
        self.params = config.quantum_params
        self.nb_params = len(self.params)

        self.simulator = JpcChip(config.quantum_parameters, config.simulation_params)

    def get_params(self):
        return self.params

    def init_params(self, quantum_params: np.ndarray) -> None:
        self.params = quantum_params

    def print_params(self) -> None:
        print(f"g_conv, g_sq: {self.params}")

    def forward(self, X: np.ndarray) -> list[Quadrature]:
        """Standard forward pass using the current nominal weights.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Output. Shape: (output_dim, batch_size).
        """

        return self.simulator.run_simulation(X, self.params[None, :])

    def forward_perturbed(self, X: np.ndarray, gradient_estimator: GradientEstimator) -> list[Quadrature]:
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
        Updates the sinus_vs_square parameters
        """
        self.params -= learning_rate * grad

