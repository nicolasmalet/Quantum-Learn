from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import accuracy

from lab.sinus_vs_square.config.data_config import batch_size, nb_epochs
from lab.sinus_vs_square.config.neural_networks import linear, first_order_adam
from lab.sinus_vs_square.config.quantum_network_config import finite_difference, zeroth_order_adam, quantum_network_config, \
    null_gradient_estimator
from quantum_learn.model import QuantumModelConfig

quantum_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=batch_size,
    nb_epochs=nb_epochs,

    neural_network_config=linear,
    neural_network_optimizer_config=first_order_adam,

    quantum_gradient_estimator=finite_difference,
    quantum_optimizer_config=zeroth_order_adam,
    quantum_network_config=quantum_network_config)

no_quantum_learning_model = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=batch_size,
    nb_epochs=nb_epochs,

    neural_network_config=linear,
    neural_network_optimizer_config=first_order_adam,

    quantum_gradient_estimator=null_gradient_estimator,
    quantum_optimizer_config=zeroth_order_adam,
    quantum_network_config=quantum_network_config)
