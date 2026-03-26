from dataclasses import dataclass

import numpy as np
from zeroth.abstract import NeuralNetworkConfig, Model, ModelConfig
from zeroth.first_order import FirstOrderOptimizer, FirstOrderNeuralNetwork, FirstOrderOptimizerConfig
from zeroth.zeroth_order import ZerothOrderOptimizer, GradientEstimator, GradientEstimatorConfig, \
    ZerothOrderOptimizerConfig
from zeroth.data import Data
from quantum_learn.quantum_black_box import QuantumBlackBox, QuantumBlackBoxConfig
from .types import BuildF


@dataclass(frozen=True)
class QuantumModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    neural_network_optimizer_config: FirstOrderOptimizerConfig

    quantum_network_config: QuantumBlackBoxConfig
    quantum_gradient_estimator: GradientEstimatorConfig
    quantum_optimizer_config: ZerothOrderOptimizerConfig

    build_f: BuildF

    def instantiate(self):
        return QuantumModel(self)


class QuantumModel(Model):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the abstract logic for training
    regardless of the underlying engine (Backpropagation or zeroth_order).
    """

    def __init__(self, config: QuantumModelConfig):

        super().__init__(config)

        self.quantum_network: QuantumBlackBox = config.quantum_network_config.instantiate()
        self.nb_quantum_params = self.quantum_network.nb_params
        self.quantum_gradient_estimator: GradientEstimator = config.quantum_gradient_estimator.instantiate(
            self.nb_quantum_params)
        self.quantum_optimizer: ZerothOrderOptimizer = config.quantum_optimizer_config.instantiate(
            self.quantum_gradient_estimator)

        self.neural_network: FirstOrderNeuralNetwork = FirstOrderNeuralNetwork(config.neural_network_config)
        self.neural_network_optimizer: FirstOrderOptimizer = config.neural_network_optimizer_config.instantiate()

        self.build_f: BuildF = config.build_f

    def train(self, data: Data, nb_print: int = 0) -> None:
        """Runs the training loop over the dataset.

        Args:
            data (Data): The dataset object containing train/test sets.
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            Array: Array of loss values recorded at each step (for plotting).
        """
        data.batch_size = self.batch_size
        nb_batches = data.nb_batches

        self.training_loss = np.empty(nb_batches, dtype=np.float64)

        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)
        print(f"    Training {self.id} Model")
        for batch_idx, (X_train, Y_train) in enumerate(data):
            X_train = X_train.ravel()

            state_history_list = self.quantum_network.forward_perturbed(X_train, self.quantum_gradient_estimator)
            perturbed_F_pred = np.stack(
                [self.build_f(state_history, self.quantum_network.simulation_constants, self.neural_network.input_dim) for state_history in state_history_list],
                axis=0)
            perturbed_Y_pred = self.neural_network.forward(perturbed_F_pred)

            avg_loss, perturbed_Loss = self.loss.compute_losses_for_zeroth_order(perturbed_Y_pred, Y_train)

            gradient = self.quantum_gradient_estimator.get_gradient(perturbed_Loss)

            F = perturbed_F_pred[0]
            self.neural_network_optimizer.do_descent(self.neural_network, self.loss, F, Y_train)
            self.quantum_optimizer.update_params(self.quantum_network, gradient)

            self.training_loss[batch_idx] = avg_loss

            if batch_idx in print_indexes:
                print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                      f"loss : {self.training_loss[batch_idx]}",
                      f"q_params : {self.quantum_network.params}")

    def test(self, data: Data) -> None:
        X_test, Y_true = data.X_test, data.Y_test
        state_history = self.quantum_network.forward(X_test)
        F_pred = self.build_f(state_history, self.quantum_network.simulation_constants, self.neural_network.input_dim)
        Y_pred = self.neural_network.forward(F_pred)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")
