from .encoding_functions import Affine
from .parameters_and_constants import JPCConfig, QuantumParameters, Encoding

jpc_config: JPCConfig = JPCConfig(
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

jpc_config_dudas: JPCConfig = JPCConfig(
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
    SIMULATION_RESOLUTION=10
)

quantum_parameters_dudas: QuantumParameters = QuantumParameters(
    g_conv=900,
    g_sq=180,
    epsilon_a=170,
    epsilon_b=170,
    delta_a=0,
    delta_b=0
)

base_encoding: Encoding = Encoding(
    encoding_parameters=("g_sq",),
    encoding_function=Affine()
)
