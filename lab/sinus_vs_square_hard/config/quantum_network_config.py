from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order.gradient_estimators import FiniteDifferenceConfig, NullGradientEstimatorConfig

from quantum_learn.quantum_black_box import QuantumBlackBoxConfig
from quantum_simulation.jpc_config import quantum_constants, simulation_constants, quantum_constants_dudas
from quantum_simulation.parameters_and_constants import QuantumParametersConfig
from .build_f import build_f_quadratures, build_f_photon_distribution

quantum_parameters = QuantumParametersConfig(g_conv=50, g_sq=20)
quantum_parameters_dudas = QuantumParametersConfig(g_conv=700, g_sq=10)
nb_chip_variables = 2

quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_constants=quantum_constants,
    quantum_parameters=quantum_parameters,
    simulation_constants=simulation_constants,
    build_F=build_f_quadratures
)

quantum_network_config_dudas = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_constants=quantum_constants_dudas,
    quantum_parameters=quantum_parameters_dudas,
    simulation_constants=simulation_constants,
    build_F=build_f_quadratures
)

quantum_photon_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_constants=quantum_constants,
    quantum_parameters=quantum_parameters,
    simulation_constants=simulation_constants,
    build_F=build_f_photon_distribution
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=1,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

finite_difference: FiniteDifferenceConfig = FiniteDifferenceConfig(dA=0.01)

null_gradient_estimator: NullGradientEstimatorConfig = NullGradientEstimatorConfig()
