import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantum_simulation import jpc_chip
from quantum_simulation.history import quadratures
from quantum_simulation.parameters_and_constants import QuantumParameters


from quantum_learn.types import Array
from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import JPCConfig




class Reservoir:
    '''
    Docstring to do.
    '''

    def __init__(self, W:jnp.array, chip:jpc_chip):
        self.W = W
        self.chip = chip
        self.F = []

    def push(self, data:jnp.array, quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        '''
        Docstring to do.
        '''
        List_state_history = self.chip.run_simulation(data, quantum_parameters_list, encoding_observable=encoding_observable)
        F = [ build_f_quadratures(List_state_history[i], simulation_constants) for i in range(len(List_state_history)) ]
        self.F = F
        return self.F


    def train(self, data:jnp.array, label:jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        '''
        Docstring.
        '''
        label_reshape =  jnp.reshape( jnp.array(label), (len(label), 1) )
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        F_inv = [ jnp.linalg.pinv(self.F[i]) for i in range(len(self.F)) ]
        self.W = [F_inv[i] @ label_reshape for i in range(len(F_inv))]

        return self.W
    
    def test(self, data:jnp.array, label_data:jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        '''Docstring to do.'''
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        Y = [ self.F[i] @ self.W[i] for i in range(len(self.W)) ]
        #plt.figure()
        #plt.plot(data, label_data, label="Target")
        #plt.plot(data,  Y[0][0])
        #plt.plot(data, Y[0][1])
        #plt.legend()
        #plt.show()
        return Y

        
        


def build_f_quadratures(state_history: StateHistory, simulation_constants: JPCConfig) -> Array:
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