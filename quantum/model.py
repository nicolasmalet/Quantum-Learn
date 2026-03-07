from zeroth.zeroth_order import ZerothOrderOptimizer, GradientEstimator, GradientEstimatorConfig, ZerothOrderOptimizerConfig
from zeroth.first_order import FirstOrderOptimizer, FirstOrderNeuralNetwork, FirstOrderOptimizerConfig
from zeroth.abstract import NeuralNetworkConfig
from zeroth import Model, ModelConfig
from .data import DataSignal

from .quantum_black_box import QuantumBlackBox, QuantumBlackBoxConfig

from dataclasses import dataclass
from matplotlib.axes import Axes
from typing import override

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class QuantumModelConfig(ModelConfig):
    """
    name (str): Name of the model (used for display and saving).
    loss (Loss): The loss class.
    metric (Callable): Function (Y_pred, Y_true) -> score (e.g., accuracy).
    batch_size (int): Number of samples per gradient update.
    plot_results (Callable): Function to visualize test results.
    nb_epochs (int): Number of passes through the entire dataset.
    """
    neural_network_config: NeuralNetworkConfig
    neural_network_optimizer_config: FirstOrderOptimizerConfig

    quantum_network_config: QuantumBlackBoxConfig
    quantum_gradient_estimator: GradientEstimatorConfig
    quantum_optimizer_config: ZerothOrderOptimizerConfig

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
        self.quantum_gradient_estimator: GradientEstimator = config.quantum_gradient_estimator.instantiate(self.nb_quantum_params)
        self.quantum_optimizer: ZerothOrderOptimizer = config.quantum_optimizer_config.instantiate(self.quantum_gradient_estimator)

        self.neural_network: FirstOrderNeuralNetwork = FirstOrderNeuralNetwork(config.neural_network_config)
        self.neural_network_optimizer: FirstOrderOptimizer = config.neural_network_optimizer_config.instantiate()

    @override
    def train(self, data: DataSignal, nb_print: int=0):
        """Runs the training loop over the dataset.

        Args:
            data (Data): The dataset object containing train/test sets.
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            np.ndarray: Array of loss values recorded at each step (for plotting).
        """
        self.print_params()
        nb_batches = data.nb_periods // self.batch_size

        self.train_loss = np.zeros(self.nb_epochs * nb_batches, dtype=np.float64)

        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)
        print(f"    Training Model")
        for epoch_idx in range(self.nb_epochs):
            print(f"        epoch n°{epoch_idx + 1} out of {self.nb_epochs}")
            data.prepare_data(self.batch_size)
            for batch_idx in range(nb_batches):
                X_train, Y_train = data.X_train[batch_idx], data.Y_train[batch_idx]

                pF_pred = self.quantum_network.forward_perturbed(X_train, self.quantum_gradient_estimator) # shape (3, out, nb_points)
                pY_pred = self.neural_network.forward(pF_pred) # shape (nb_params + 1, out, nb_points)
                avg_loss, pLoss = self.loss.compute_losses_for_zeroth_order(pY_pred, Y_train)

                gradient = self.quantum_gradient_estimator.get_gradient(pLoss)

                F = pF_pred[0]
                self.neural_network_optimizer.do_descent(self.neural_network, self.loss, F, Y_train)
                self.quantum_optimizer.update_params(self.quantum_network, gradient)

                self.train_loss[epoch_idx * nb_batches + batch_idx] = avg_loss

                if batch_idx in print_indexes:
                    print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                          f"loss : {np.round(self.train_loss[epoch_idx * nb_batches + batch_idx], 3)}")
                    self.print_params()

    @override
    def test(self, data):
        X_test, Y_true = data.X_test, data.Y_test
        F_pred = self.quantum_network.forward(X_test)[0, :, :]
        Y_pred = self.neural_network.forward(F_pred)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")

    def print_params(self) -> None:
        self.quantum_network.print_params()
        self.neural_network.print_params()