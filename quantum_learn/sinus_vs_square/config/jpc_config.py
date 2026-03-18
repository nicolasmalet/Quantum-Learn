from quantum_simulation.quantum_params import QuantumParams
from quantum_simulation.simulation_params import SimulationParams
from .data_config import MEASURE_RESOLUTION

NB_QUADRATURES = 4

quantum_params = QuantumParams(
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

simulation_params = SimulationParams(MEASURE_RESOLUTION=MEASURE_RESOLUTION,
                                     SIMULATION_RESOLUTION=MEASURE_RESOLUTION)
