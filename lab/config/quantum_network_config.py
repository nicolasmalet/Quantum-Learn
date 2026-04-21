from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order import gradient_estimators as g_e
from zeroth.utils.perturbation_matrices import RademacherMatrix
from quantum_learn import build_f
from quantum_learn.quantum_black_box import QuantumBlackBoxConfig
from quantum_simulation.configs import jpc_config_dudas, quantum_parameters_dudas, base_encoding
from quantum_simulation.parameters_and_constants import QuantumParameters

network_quad = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    encoding=base_encoding,
    quantum_parameters=quantum_parameters_dudas,
    build_f_config=build_f.BuildFQuadraturesConfig()
)

network_quad_poly = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    encoding=base_encoding,
    quantum_parameters=quantum_parameters_dudas,
    build_f_config=build_f.BuildFQuadraturesPolynomialsConfig()
)

network_photon = QuantumBlackBoxConfig(
    name="Q_Network",
    jpc_config=jpc_config_dudas,
    encoding=base_encoding,
    quantum_parameters=quantum_parameters_dudas,
    build_f_config=build_f.BuildFProbasConfig(clip_probas=2)
)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=0.02,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

global_finite_difference: g_e.GlobalFiniteDifferenceConfig = g_e.GlobalFiniteDifferenceConfig(dA=0.1)

partial_gs_finite_difference: g_e.PartialFiniteDifferenceConfig = g_e.PartialFiniteDifferenceConfig(dA=0.1,
                                                                                            indexes=QuantumParameters.get_indices(
                                                                                                "g_conv"))

null_gradient_estimator: g_e.NullGradientEstimatorConfig = g_e.NullGradientEstimatorConfig()

simultaneous_gradient_estimator = g_e.SimultaneousPerturbationConfig = g_e.SimultaneousPerturbationConfig(dA = 0.1,
                                                                                              nb_perturbations=2,
                                                                                              get_perturbation_matrix=RademacherMatrix()
                                                                                              )