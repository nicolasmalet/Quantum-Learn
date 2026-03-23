from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import accuracy

from quantum_learn.model import QuantumModelConfig
from .config.data_config import batch_size, nb_epochs
from .config import neural_networks as nn
from .config import quantum_network_config as qn


quantum_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=batch_size,
    nb_epochs=nb_epochs,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_gradient_estimator=qn.finite_difference,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_network_config=qn.quantum_network_config)

no_quantum_learning_model = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=batch_size,
    nb_epochs=nb_epochs,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_gradient_estimator=qn.null_gradient_estimator,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_network_config=qn.quantum_network_config)

photon_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=batch_size,
    nb_epochs=nb_epochs,

    neural_network_config=nn.linear_photons,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_gradient_estimator=qn.finite_difference,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_network_config=qn.quantum_photon_config)