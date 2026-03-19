import numpy as np
from zeroth.zeroth_order import ZerothOrderAdamConfig, GradientEstimatorConfig, GradientEstimator
from zeroth.zeroth_order.gradient_estimators import FiniteDifferenceConfig

from .jpc_config import quantum_params, simulation_params
from .. null_gradient_estimator import NullGradientEstimatorConfig
from ...quantum_black_box import QuantumBlackBoxConfig


CHIP_VARIABLES = np.array([50.0, 50.0])
NB_CHIP_VARIABLES = 2


quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_params=CHIP_VARIABLES,
    quantum_parameters=quantum_params,
    simulation_params=simulation_params,
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=0.02,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

finite_difference: FiniteDifferenceConfig = FiniteDifferenceConfig(dA=0.01)

null_gradient_estimator: NullGradientEstimatorConfig = NullGradientEstimatorConfig(dA=1)