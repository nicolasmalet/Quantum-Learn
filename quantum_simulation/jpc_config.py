from quantum_simulation.parameters_and_constants.quantum_constants import QuantumConstants
from quantum_simulation.parameters_and_constants.simulation_constants import SimulationConstants
from lab.sinus_vs_square_hard.config.data_config import measure_resolution, simulation_resolution

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

simulation_constants = SimulationConstants(MEASURE_RESOLUTION=measure_resolution,
                                           SIMULATION_RESOLUTION=simulation_resolution)
