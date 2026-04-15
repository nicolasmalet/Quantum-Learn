from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order.gradient_estimators import PartialFiniteDifferenceConfig, GlobalFiniteDifferenceConfig, \
    NullGradientEstimatorConfig

from quantum_learn import build_f
from quantum_learn.quantum_black_box import QuantumBlackBoxConfig
from quantum_simulation.configs import jpc_config, jpc_config_dudas, quantum_parameters_config
from quantum_simulation.parameters_and_constants import QuantumParametersConfig

quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config,
    quantum_parameters=quantum_parameters_config,
    build_f_config=build_f.BuildFQuadraturesConfig()
)

quantum_network_config_dudas = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    quantum_parameters=quantum_parameters_config,
    build_f_config=build_f.BuildFQuadraturesConfig()
)

quantum_photon_config = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    quantum_parameters=quantum_parameters_config,
    build_f_config=build_f.BuildFPhotonDistributionConfig()
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=1,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

global_finite_difference: GlobalFiniteDifferenceConfig = GlobalFiniteDifferenceConfig(dA=0.1)

partial_gs_finite_difference: PartialFiniteDifferenceConfig = PartialFiniteDifferenceConfig(dA=0.1,
                                                                                            indexes=QuantumParametersConfig.get_indices(
                                                                                                "g_conv"))

null_gradient_estimator: NullGradientEstimatorConfig = NullGradientEstimatorConfig()
