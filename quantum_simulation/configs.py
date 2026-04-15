from .parameters_and_constants import QuantumParameters
from .parameters_and_constants.jpc_config import JPCConfig

jpc_config = JPCConfig(
    DIM_A=15,
    DIM_B=15,
    OMEGA_A=1e4,
    OMEGA_B=9 * 1e3,
    KAPPA_A=17,
    KAPPA_B=21,
    K_AA=0.1,
    K_BB=0.1,
    K_AB=0.05,
    DRIVE_DURATION=0.05,
    MEASURE_RESOLUTION=10,
    SIMULATION_RESOLUTION=10
)

jpc_config_dudas = JPCConfig(
    DIM_A=10,
    DIM_B=10,
    OMEGA_A=1e4,
    OMEGA_B=9 * 1e3,
    KAPPA_A=17,
    KAPPA_B=21,
    K_AA=0.1,
    K_BB=0.1,
    K_AB=0.05,
    DRIVE_DURATION=0.04,
    MEASURE_RESOLUTION=10,
    SIMULATION_RESOLUTION=100
)

quantum_parameters_config = QuantumParameters(g_conv=700, g_sq=70, epsilon_a=65, epsilon_b=65, delta_a=10,
                                              delta_b=12)
