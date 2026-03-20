from dataclasses import dataclass, fields

import numpy as np
from zeroth.zeroth_order.gradient_estimators import GradientEstimator
from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox

from quantum_simulation.jpc_chip import JpcChip
from quantum_simulation.parameters_and_constants.quantum_constants import QuantumConstants
from quantum_simulation.parameters_and_constants.quantum_parameters import QuantumParameters
from quantum_simulation.parameters_and_constants.simulation_constants import SimulationConstants
from quantum_learn.types import BuildF


@dataclass
class QuantumBlackBoxConfig:
    name: str
    quantum_constants: QuantumConstants
    quantum_parameters: QuantumParameters
    simulation_constants: SimulationConstants
    build_F: BuildF

    def instantiate(self):
        return QuantumBlackBox(self)


class QuantumBlackBox(ZerothOrderBlackBox):
    def __init__(self, config: QuantumBlackBoxConfig) -> None:
        self.name: str = config.name
        self.params: QuantumParameters = config.quantum_parameters
        self.nb_params: int = len(fields(self.params))

        self.quantum_constants = config.quantum_constants
        self.simulation_constants = config.simulation_constants

        self.simulator: JpcChip = JpcChip(config.quantum_constants, config.simulation_constants)
        self.build_F: BuildF = config.build_F

    def get_params(self) -> QuantumParameters:
        return self.params

    def init_params(self, quantum_params: QuantumParameters) -> None:
        self.params = quantum_params

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Standard forward pass using the current nominal weights.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Output. Shape: (output_dim, batch_size).
        """
        state_history = self.simulator.run_simulation(X, [self.params])[0]
        return self.build_F(state_history, self.simulation_constants)

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
        perturbed_params = [QuantumParameters(*params) for params in gradient_estimator.perturb(self.params.as_array())]
        state_history_list = self.simulator.run_simulation(X, perturbed_params)
        return np.stack([self.build_F(state_history, self.simulation_constants) for state_history in state_history_list], axis=0)

    def update_params(self, grad: np.ndarray, learning_rate: float) -> None:
        """
        Updates the sinus_vs_square parameters
        """
        self.params -= learning_rate * grad
