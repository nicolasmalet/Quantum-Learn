from zeroth.losses import MSE
from zeroth.losses.binary_cross_entropy import BinaryCrossEntropy
from zeroth.utils.metrics import Accuracy

from ..config import quantum_network_config as qn, neural_networks as nn
from quantum_learn.model import QuantumModelConfig
from quantum_learn.reservoir_model import ReservoirModelConfig

batch_size = 8 * 10

reservoir_model_config = ReservoirModelConfig(
    name="Quantum Model",
    id={},
    loss=MSE(),
    metric=Accuracy(),
    batch_size = batch_size,

    quantum_blackbox_config=qn.network_quad,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.simultaneous_gradient_estimator
)

base_model = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_blackbox_config=qn.network_quad,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.global_finite_difference
)

no_quantum_learning_model = QuantumModelConfig(
    name="No quantum Learning Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_blackbox_config=qn.network_quad,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.null_gradient_estimator
)

quantum_model_config_dudas_train_all = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_blackbox_config=qn.network_quad,
    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_gradient_estimator=qn.global_finite_difference
)

quantum_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.network_quad,
    quantum_gradient_estimator=qn.global_finite_difference
)

no_quantum_learning_model_xs = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.XS,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.network_quad,
    quantum_gradient_estimator=qn.null_gradient_estimator,
)

photon_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=BinaryCrossEntropy(),
    metric=Accuracy(),
    batch_size=batch_size,

    neural_network_config=nn.linear,
    neural_network_optimizer_config=nn.first_order_adam,

    quantum_optimizer_config=qn.zeroth_order_adam,
    quantum_blackbox_config=qn.network_photon,
    quantum_gradient_estimator=qn.partial_gs_finite_difference
)
