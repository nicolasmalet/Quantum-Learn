from dataclasses import dataclass

import numpy as np
from zeroth import abstract, first_order
from zeroth import zeroth_order
from zeroth.data import Data

from quantum_learn.quantum_black_box import QuantumBlackBox, QuantumBlackBoxConfig


@dataclass(frozen=True)
class QuantumModelConfig(abstract.ModelConfig):
    neural_network_config: abstract.NeuralNetworkConfig
    neural_network_optimizer_config: first_order.FirstOrderOptimizerConfig

    quantum_blackbox_config: QuantumBlackBoxConfig
    quantum_gradient_estimator: zeroth_order.GradientEstimatorConfig
    quantum_optimizer_config: zeroth_order.ZerothOrderOptimizerConfig

    def instantiate(self, data: Data):
        return QuantumModel(self, data)


class QuantumModel(abstract.Model):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the abstract logic for training
    regardless of the underlying engine (Backpropagation or zeroth_order).
    """

    def __init__(self, config: QuantumModelConfig, data: Data):

        super().__init__(config, data)

        self.quantum_blackbox: QuantumBlackBox = config.quantum_blackbox_config.instantiate()
        self.nb_quantum_params = self.quantum_blackbox.nb_params
        self.quantum_gradient_estimator: zeroth_order.GradientEstimator = config.quantum_gradient_estimator.instantiate(
            self.nb_quantum_params)
        self.quantum_optimizer: zeroth_order.ZerothOrderOptimizer = config.quantum_optimizer_config.instantiate(
            self.quantum_gradient_estimator)

        self.neural_network: first_order.FirstOrderNeuralNetwork = first_order.FirstOrderNeuralNetwork(
            config=config.neural_network_config, input_dim=self.quantum_blackbox.output_dim,
            output_dim=self.data.output_dim)
        self.neural_network_optimizer: first_order.FirstOrderOptimizer = config.neural_network_optimizer_config.instantiate()

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
            X_train = X_train.ravel()

            perturbed_F_pred = self.quantum_blackbox.forward_perturbed(X_train, self.quantum_gradient_estimator)
            perturbed_Y_pred = self.neural_network(perturbed_F_pred)

            avg_loss, perturbed_Loss = self.loss.compute_losses_for_zeroth_order(perturbed_Y_pred, Y_train)

            gradient = self.quantum_gradient_estimator.get_gradient(perturbed_Loss)

            F = perturbed_F_pred[0]

            for _ in range(100):
                self.neural_network_optimizer.do_descent(self.neural_network, self.loss, F, Y_train)

            self.quantum_optimizer.update_params(self.quantum_blackbox, gradient)

            self.training_loss[batch_idx] = avg_loss

            if batch_idx in print_indexes:
                print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                      f"loss : {self.training_loss[batch_idx]}",
                      f"q_params : {self.quantum_blackbox.params}")

    def test(self) -> None:
        X_test, Y_true = self.data.X_test, self.data.Y_test
        F_pred = self.quantum_blackbox(X_test)
        Y_pred = self.neural_network.forward(F_pred)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")
