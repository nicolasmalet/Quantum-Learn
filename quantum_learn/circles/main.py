from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import accuracy

from .configs import *
from .data import create_data_circle
from .model import QuantumModelConfig

Q_ModelConfig = QuantumModelConfig(
    name="Q_Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=10,
    nb_epochs=1,

    neural_network_config=linear,
    neural_network_optimizer_config=first_order_adam,

    quantum_gradient_estimator=finite_difference,
    quantum_optimizer_config=zeroth_order_adam,
    quantum_network_config=quantum_network_config)


def main():
    M = Q_ModelConfig.instantiate()
    data = create_data_circle(1000, 100, 0.7)
    M.train(data, 100)
    M.plot_loss()
    M.test(data)
