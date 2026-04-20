from dataclasses import dataclass
from typing import override

import numpy as np
from zeroth import abstract
from zeroth import zeroth_order
from zeroth.data import Data

from quantum_learn.quantum_black_box import QuantumBlackBox, QuantumBlackBoxConfig
from .reservoir import Reservoir
from .types import Array


@dataclass(frozen=True)
class ReservoirModelConfig(abstract.ModelConfig):
    quantum_blackbox_config: QuantumBlackBoxConfig
    quantum_gradient_estimator: zeroth_order.GradientEstimatorConfig
    quantum_optimizer_config: zeroth_order.ZerothOrderOptimizerConfig

    def instantiate(self, data: Data):
        return ReservoirModel(self, data)


class ReservoirModel(abstract.Model):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the abstract logic for training
    regardless of the underlying engine (Backpropagation or zeroth_order).
    """

    def __init__(self, config: ReservoirModelConfig, data: Data):

        super().__init__(config, data)

        self.quantum_blackbox: QuantumBlackBox = config.quantum_blackbox_config.instantiate()
        self.nb_quantum_params = self.quantum_blackbox.nb_params
        self.quantum_gradient_estimator: zeroth_order.GradientEstimator = config.quantum_gradient_estimator.instantiate(
            self.nb_quantum_params)
        self.quantum_optimizer: zeroth_order.ZerothOrderOptimizer = config.quantum_optimizer_config.instantiate(
            self.quantum_gradient_estimator)

        self.neural_network: Reservoir = Reservoir()

    @override
    def train(self, nb_print: int = 0) -> None:
        """Runs the training loop over the dataset.

        Args:
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            Array: Array of loss values recorded at each step (for plotting).
        """
        self.data.batch_size = self.batch_size
        nb_batches = len(self.data)

        self.training_loss = np.empty(nb_batches, dtype=np.float64)

        nb_print = nb_batches if nb_print == -1 else nb_print
        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)

        print(f"    Training {self.id} Model")
        for batch_idx, (X_train, Y_train) in enumerate(self.data):

            loss = self._training_step(X_train, Y_train)
            self.training_loss[batch_idx] = loss

            if batch_idx in print_indexes:
                print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                      f"loss : {loss}",
                      f"q_params : {self.quantum_blackbox.params}")

    def _training_step(self, X_train: Array, Y_train: Array) -> float:
        X_train = X_train.ravel()
        perturbed_F_pred = self.quantum_blackbox.forward_perturbed(X_train, self.quantum_gradient_estimator)
        F = perturbed_F_pred[0]

        new_W = self.neural_network.compute_W(F, Y_train)
        self.neural_network.W = self.neural_network.W * 0.9 + 0.1 * new_W

        print("Perfect W, terrible params")
        self.quick_test()
        perturbed_Y_pred = self.neural_network(perturbed_F_pred)
        avg_loss, perturbed_loss = self.loss.compute_losses_for_zeroth_order(perturbed_Y_pred, Y_train)


        gradient = self.quantum_gradient_estimator.get_gradient(perturbed_loss)
        self.quantum_optimizer.update_params(self.quantum_blackbox, gradient)
        print("Perfect params terrible W")
        self.quick_test()
        return avg_loss

    @override
    def test(self) -> None:
        X_test, Y_true = self.data.X_test, self.data.Y_test
        F_pred = self.quantum_blackbox(X_test)
        Y_pred = self.neural_network.forward(F_pred)


        self.test_accuracy = float(np.mean(Y_pred == Y_true))
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")

    def quick_test(self):
        X_test = np.array([-0.7, 0.0, 0.7, 1.0, 0.7, -0.0, -0.7, -1.0,
                           1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        Y_test = np.array([[1], [1], [1], [1], [1], [1], [1], [1],
                           [0], [0], [0], [0], [0], [0], [0], [0]])
        F = self.quantum_blackbox(X_test)
        Y_pred = self.neural_network(F)
        print(f"loss : {self.loss.compute_loss(Y_pred, Y_test)}")
