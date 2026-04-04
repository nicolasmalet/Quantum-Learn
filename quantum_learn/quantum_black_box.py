from dataclasses import dataclass, fields

import numpy as np
from zeroth.zeroth_order.gradient_estimators import GradientEstimator
from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox

from quantum_simulation.jpc_chip import JpcChip
from quantum_simulation.parameters_and_constants.jpc_config import JPCConfig
from quantum_simulation.parameters_and_constants.quantum_parameters import QuantumParameters, QuantumParametersConfig
from .build_f import BuildF, BuildFConfig
from .types import Array


@dataclass
class QuantumBlackBoxConfig:
    name: str
    jpc_config: JPCConfig
    quantum_parameters: QuantumParametersConfig
    build_f_config: BuildFConfig

    def instantiate(self):
        return QuantumBlackBox(self)


class QuantumBlackBox(ZerothOrderBlackBox):
    def __init__(self, config: QuantumBlackBoxConfig) -> None:
        self.name: str = config.name
        self.params: QuantumParameters = config.quantum_parameters.instantiate()
        self.nb_params: int = len(fields(self.params))

        self.jpc_config = config.jpc_config

        self.simulator: JpcChip = JpcChip(config.jpc_config)

        self.build_f: BuildF = config.build_f_config.instantiate(config.jpc_config)
        self.output_dim = self.build_f.output_dim

    def __call__(self, X: Array) -> Array:
        return self.forward(X)

    def get_params(self) -> QuantumParameters:
        return self.params

    def init_params(self, quantum_params: QuantumParameters) -> None:
        self.params = quantum_params

    def forward(self, X: Array) -> Array:
        """Standard forward pass using the current nominal weights.

        Args:
            X (Array): Input batch. Shape: (input_dim, batch_size).

        Returns:
            Array: Output. Shape: (output_dim, batch_size).
        """
        state_history = self.simulator.run_simulation(X, [self.params])[0]

        F = self.build_f(state_history)
        return F

    def forward_perturbed(self, X: Array, gradient_estimator: GradientEstimator) -> Array:
        """Parallel forward pass for multiple perturbed versions of the network.

        This method broadcasts the input X across T perturbed parameter sets
        to compute T outputs simultaneously without a Python loop.

        Args:
            X (Array): Input batch. Shape: (input_dim, batch_size).
            gradient_estimator (GradientEstimator): The gradient_estimator object.
        """
        perturbed_params = [QuantumParameters(*params) for params in gradient_estimator.perturb(self.params.as_array())]
        state_history_list = self.simulator.run_simulation(X, perturbed_params)
        F_pred_list = [self.build_f(state_history) for state_history in state_history_list]
        perturbed_F_pred = np.stack(F_pred_list, axis=0)

        return perturbed_F_pred

    def update_params(self, grad: Array, learning_rate: float) -> None:
        """
        Updates the sinus_vs_square parameters
        """
        self.params -= learning_rate * grad
