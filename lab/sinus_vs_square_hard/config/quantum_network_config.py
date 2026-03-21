from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order.gradient_estimators import FiniteDifferenceConfig, NullGradientEstimatorConfig

from quantum_simulation.parameters_and_constants import QuantumParameters
from .build_f import build_F_Quadratures
from quantum_simulation.jpc_config import quantum_constants, simulation_constants
from quantum_learn.quantum_black_box import QuantumBlackBoxConfig

quantum_parameters = QuantumParameters(g_conv=50, g_sq=20)
nb_chip_variables = 2

quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_constants=quantum_constants,
    quantum_parameters=quantum_parameters,
    simulation_constants=simulation_constants,
    build_F=build_F_Quadratures
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=1,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

finite_difference: FiniteDifferenceConfig = FiniteDifferenceConfig(dA=0.01)

null_gradient_estimator: NullGradientEstimatorConfig = NullGradientEstimatorConfig()
