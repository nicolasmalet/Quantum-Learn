import numpy as np

from quantum_learn.types import Array
from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import SimulationConstants


def build_f_quadratures(state_history: StateHistory, simulation_constants: SimulationConstants) -> Array:
    """
    Construit la feature matrix selon la bonne notation

    Returns
    -------
    F : Array
        Feature matrix F(X)
    """

    nb_quadratures = 4

    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION
    input_dim = nb_quadratures * simulation_constants.MEASURE_RESOLUTION

    L_Ia = state_history.quadratures.L_Ia[::step]
    L_Qa = state_history.quadratures.L_Qa[::step]
    L_Ib = state_history.quadratures.L_Ib[::step]
    L_Qb = state_history.quadratures.L_Qb[::step]

    L_Ia = L_Ia.reshape(-1, input_dim // nb_quadratures).T
    L_Qa = L_Qa.reshape(-1, input_dim // nb_quadratures).T
    L_Ib = L_Ib.reshape(-1, input_dim // nb_quadratures).T
    L_Qb = L_Qb.reshape(-1, input_dim // nb_quadratures).T

    return np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))


def build_f_photon_distribution(state_history: StateHistory, simulation_constants: SimulationConstants) -> Array:
    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION
    measures = state_history.photon_distribution.joint_proba[::step, :, :]
    nb_measures, dim1, dim2 = measures.shape
    F = measures.reshape(-1, simulation_constants.MEASURE_RESOLUTION * dim1 * dim2).T
    return F