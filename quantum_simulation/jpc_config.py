from .parameters_and_constants.quantum_constants import QuantumConstants
from .parameters_and_constants.simulation_constants import SimulationConstants

nb_quadratures = 4

quantum_constants = QuantumConstants(
    DIM_A=15,
    DIM_B=15,
    OMEGA_A=1e4,
    OMEGA_B=9 * 1e3,
    KAPPA_A=17,
    KAPPA_B=21,
    K_AA=0.1,
    K_BB=0.1,
    K_AB=0.05,
    EPSILON_A=20,
    EPSILON_B=20,
    DRIVE_DURATION=0.05
)

quantum_constants_dudas = QuantumConstants(
    DIM_A=15,
    DIM_B=15,
    OMEGA_A=1e4,
    OMEGA_B=9 * 1e3,
    KAPPA_A=17,
    KAPPA_B=21,
    K_AA=0.1,
    K_BB=0.1,
    K_AB=0.05,
    EPSILON_A=130,
    EPSILON_B=130,
    DRIVE_DURATION=0.04
)



simulation_constants = SimulationConstants(MEASURE_RESOLUTION=10,
                                           SIMULATION_RESOLUTION=10)
