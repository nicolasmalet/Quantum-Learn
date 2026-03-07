from .model import QuantumModelConfig
from .data import create_data_sinus_vs_square
from .configs import *

from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import accuracy


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
    data = create_data_sinus_vs_square(500, 100)
    M.train(data, 50)
    M.plot_loss()
    M.test(data)
