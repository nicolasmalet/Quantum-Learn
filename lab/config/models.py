from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import Accuracy

from lab.config import quantum_network_config as qn, neural_networks as nn
from quantum_learn.model import QuantumModelConfig

batch_size = 10 * 8

quantum_model_config_dudas = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_blackbox_config=qn.quantum_network_config_dudas,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.global_finite_difference
)

quantum_model_config_dudas_train_all = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_blackbox_config=qn.quantum_network_config_dudas,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.global_finite_difference
)

quantum_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.quantum_network_config,
    quantum_gradient_estimator=qn.global_finite_difference
)

no_quantum_learning_model_xs = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.quantum_network_config,
    quantum_gradient_estimator=qn.null_gradient_estimator,
)

photon_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.quantum_photon_config,
    quantum_gradient_estimator=qn.partial_gs_finite_difference
)
