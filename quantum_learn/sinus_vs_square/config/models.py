from zeroth.losses import CrossEntropy
from zeroth.utils.metrics import accuracy

from .data_config import BATCH_SIZE, NB_EPOCHS
from .neural_networks import linear, first_order_adam
from .quantum_network_config import finite_difference, zeroth_order_adam, quantum_network_config
from ..model import QuantumModelConfig

quantum_model_config = QuantumModelConfig(
    name="Quantum Model",
    id={},
    loss=CrossEntropy(),
    metric=accuracy,
    batch_size=BATCH_SIZE,
    nb_epochs=NB_EPOCHS,

    neural_network_config=linear,
    neural_network_optimizer_config=first_order_adam,

    quantum_gradient_estimator=finite_difference,
    quantum_optimizer_config=zeroth_order_adam,
    quantum_network_config=quantum_network_config)
