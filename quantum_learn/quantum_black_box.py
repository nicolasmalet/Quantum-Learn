from dataclasses import dataclass, fields

from zeroth.zeroth_order.gradient_estimators import GradientEstimator
from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox

from quantum_simulation.history import StateHistory
from quantum_simulation.jpc_chip import JpcChip
from quantum_simulation.parameters_and_constants.quantum_constants import QuantumConstants
from quantum_simulation.parameters_and_constants.quantum_parameters import QuantumParameters, QuantumParametersConfig
from quantum_simulation.parameters_and_constants.simulation_constants import SimulationConstants
from .types import Array


@dataclass
class QuantumBlackBoxConfig:
    name: str
    quantum_constants: QuantumConstants
    quantum_parameters: QuantumParametersConfig
    simulation_constants: SimulationConstants

    def instantiate(self):
        return QuantumBlackBox(self)


class QuantumBlackBox(ZerothOrderBlackBox):
    def __init__(self, config: QuantumBlackBoxConfig) -> None:
        self.name: str = config.name
        self.params: QuantumParameters = config.quantum_parameters.instantiate()
        self.nb_params: int = len(fields(self.params))

        self.quantum_constants = config.quantum_constants
        self.simulation_constants = config.simulation_constants

        self.simulator: JpcChip = JpcChip(config.quantum_constants, config.simulation_constants)

    def get_params(self) -> QuantumParameters:
        return self.params

    def init_params(self, quantum_params: QuantumParameters) -> None:
        self.params = quantum_params

    def forward(self, X: Array) -> StateHistory:
        """Standard forward pass using the current nominal weights.

        Args:
            X (Array): Input batch. Shape: (input_dim, batch_size).

        Returns:
            Array: Output. Shape: (output_dim, batch_size).
        """
        return self.simulator.run_simulation(X, [self.params])[0]

    def forward_perturbed(self, X: Array, gradient_estimator: GradientEstimator) -> list[StateHistory]:
        """Parallel forward pass for multiple perturbed versions of the network.

        This method broadcasts the input X across T perturbed parameter sets
        to compute T outputs simultaneously without a Python loop.

        Args:
            X (Array): Input batch. Shape: (input_dim, batch_size).
            gradient_estimator (GradientEstimator): The gradient_estimator object.
        """
        perturbed_params = [QuantumParameters(*params) for params in gradient_estimator.perturb(self.params.as_array())]
        return self.simulator.run_simulation(X, perturbed_params)

    def update_params(self, grad: Array, learning_rate: float) -> None:
        """
        Updates the sinus_vs_square parameters
        """
        self.params -= learning_rate * grad
