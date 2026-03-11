
from zeroth.utils.metrics import accuracy
from zeroth.losses import CrossEntropy

from ..model import QuantumModelConfig
from .neural_network_config import linear, first_order_adam
from .black_box_config import finite_difference, zeroth_order_adam, quantum_network_config


quantum_model_config = QuantumModelConfig(
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

