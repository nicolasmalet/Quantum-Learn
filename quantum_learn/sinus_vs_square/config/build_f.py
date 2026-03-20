import numpy as np

from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import SimulationConstants


def build_F_Quadratures(state_history: StateHistory, simulation_constants: SimulationConstants,
                        input_dim: int) -> np.ndarray:
    """
    Construit la feature matrix selon la bonne notation

    Returns
    -------
    F : np.ndarray
        Feature matrix F(X)
    """
    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION

    L_Ia = state_history.quadratures.L_Ia[::step]
    L_Qa = state_history.quadratures.L_Qa[::step]
    L_Ib = state_history.quadratures.L_Ib[::step]
    L_Qb = state_history.quadratures.L_Qb[::step]

    L_Ia = L_Ia.reshape(-1, input_dim).T
    L_Qa = L_Qa.reshape(-1, input_dim).T
    L_Ib = L_Ib.reshape(-1, input_dim).T
    L_Qb = L_Qb.reshape(-1, input_dim).T

    return np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))
