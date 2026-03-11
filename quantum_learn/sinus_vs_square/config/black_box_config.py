from zeroth.zeroth_order.gradient_estimators import FiniteDifferenceConfig
from zeroth.zeroth_order import ZerothOrderAdamConfig

from ...quantum_black_box import QuantumBlackBoxConfig
from .jpc_config import quantum_params, simulation_params

import numpy as np


quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_params=np.array([50.0, 50.0]),
    quantum_parameters=quantum_params,
    simulation_params=simulation_params,
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=0.02,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

finite_difference: FiniteDifferenceConfig = FiniteDifferenceConfig(dA=0.01)

