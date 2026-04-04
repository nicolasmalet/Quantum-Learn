from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order.gradient_estimators import PartialFiniteDifferenceConfig, GlobalFiniteDifferenceConfig, \
    NullGradientEstimatorConfig

from quantum_learn.quantum_black_box import QuantumBlackBoxConfig
from quantum_simulation.configs import jpc_config, jpc_config_dudas
from quantum_simulation.parameters_and_constants import QuantumParametersConfig
from quantum_learn import build_f

quantum_parameters = QuantumParametersConfig(g_conv_real=50, g_conv_imag=0, g_sq_real=20, g_sq_imag=0, delta_a=0,
                                             delta_b=0)
quantum_parameters_dudas = QuantumParametersConfig(g_conv_real=700, g_conv_imag=0, g_sq_real=10, g_sq_imag=0, delta_a=0,
                                                   delta_b=0)

nb_chip_variables = 2

quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config,
    quantum_parameters=quantum_parameters,
    build_f_config=build_f.BuildFQuadraturesConfig()
)

quantum_network_config_dudas = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    quantum_parameters=quantum_parameters_dudas,
    build_f_config=build_f.BuildFQuadraturesConfig()
)

quantum_photon_config = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config,
    quantum_parameters=quantum_parameters,
    build_f_config=build_f.BuildFPhotonDistributionConfig()
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=1,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

global_finite_difference: GlobalFiniteDifferenceConfig = GlobalFiniteDifferenceConfig(dA=0.01)

partial_gs_finite_difference: PartialFiniteDifferenceConfig = PartialFiniteDifferenceConfig(dA=0.01,
                                                                                            indexes=QuantumParametersConfig.get_indices("g_conv_real", "g_sq_real"))

null_gradient_estimator: NullGradientEstimatorConfig = NullGradientEstimatorConfig()
