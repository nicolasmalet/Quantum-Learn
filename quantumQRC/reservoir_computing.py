import jax.numpy as jnp

from quantum_simulation.history import StateHistory
from quantum_simulation.jpc_chip import JpcChip
from quantum_simulation.parameters_and_constants import JPCConfig
from quantum_simulation.parameters_and_constants import QuantumParameters


class Reservoir:
    def __init__(self, W: jnp.ndarray, chip: JpcChip) -> None:
        self.W = W
        self.chip: JpcChip = chip
        self.F: list = []

    def push(self, data: jnp.ndarray, quantum_parameters_list: list[QuantumParameters],
             simulation_constants: JPCConfig):
        List_state_history = self.chip.run_simulation(data, quantum_parameters_list)
        F = [build_f_quadratures(List_state_history[i], simulation_constants) for i in range(len(List_state_history))]
        self.F = F
        return self.F

    def train(self, data: jnp.ndarray, label: jnp.ndarray,
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig):
        label_reshape = jnp.reshape(jnp.array(label), (len(label), 1))
        self.push(data, quantum_parameters_list, simulation_constants)
        F_inv = [jnp.linalg.pinv(self.F[i]) for i in range(len(self.F))]
        self.W = [F_inv[i] @ label_reshape for i in range(len(F_inv))]

        return self.W

    def test(self, data: jnp.ndarray, label_data: jnp.ndarray,
             quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig):
        self.push(data, quantum_parameters_list, simulation_constants)
        Y = [self.F[i] @ self.W[i] for i in range(len(self.W))]
        # plt.figure()
        # plt.plot(data, label_data, label="Target")
        # plt.plot(data,  Y[0][0])
        # plt.plot(data, Y[0][1])
        # plt.legend()
        # plt.show()
        return Y


def build_f_quadratures(state_history: StateHistory, simulation_constants: JPCConfig) -> jnp.ndarray:
    """
    Construit la feature matrix

    Returns
    -------
    F : Array
        Feature matrix F(X)
    """
    nb_quadratures = 4
    input_dim = nb_quadratures * simulation_constants.MEASURE_RESOLUTION

    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION

    L_Ia = state_history.quadratures.L_Ia[::step]
    L_Qa = state_history.quadratures.L_Qa[::step]
    L_Ib = state_history.quadratures.L_Ib[::step]
    L_Qb = state_history.quadratures.L_Qb[::step]

    L_Ia = L_Ia.reshape(-1, input_dim // nb_quadratures)
    L_Qa = L_Qa.reshape(-1, input_dim // nb_quadratures)
    L_Ib = L_Ib.reshape(-1, input_dim // nb_quadratures)
    L_Qb = L_Qb.reshape(-1, input_dim // nb_quadratures)

    F = jnp.hstack((L_Ia, L_Qa, L_Ib, L_Qb))

    return F
