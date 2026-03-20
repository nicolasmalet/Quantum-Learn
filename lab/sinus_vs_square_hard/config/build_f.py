import numpy as np

from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import SimulationConstants
from quantum_simulation.jpc_config import nb_quadratures
from quantum_learn.types import Array




def build_F_Quadratures(state_history: StateHistory, simulation_constants: SimulationConstants) -> Array:
    """
    Construit la feature matrix selon la bonne notation

    Returns
    -------
    F : Array
        Feature matrix F(X)
    """
    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION
    input_dim = nb_quadratures * simulation_constants.MEASURE_RESOLUTION

    L_Ia = state_history.quadratures.L_Ia[::step]
    L_Qa = state_history.quadratures.L_Qa[::step]
    L_Ib = state_history.quadratures.L_Ib[::step]
    L_Qb = state_history.quadratures.L_Qb[::step]

    L_Ia = L_Ia.reshape(-1, input_dim // 4).T
    L_Qa = L_Qa.reshape(-1, input_dim // 4).T
    L_Ib = L_Ib.reshape(-1, input_dim // 4).T
    L_Qb = L_Qb.reshape(-1, input_dim // 4).T

    return np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))
