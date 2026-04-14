import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantum_simulation import jpc_chip
from quantum_simulation.history import quadratures
from quantum_simulation.parameters_and_constants import QuantumParameters


from quantum_learn.types import Array
from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import JPCConfig

import equinox as eqx

@eqx.filter_jit
def _jit_compute_weights(F_stack, label_reshape):
    """Calcule toutes les pseudo-inverses et les poids d'un coup (Batching)."""
    F_inv = jnp.linalg.pinv(F_stack)
    return F_inv @ label_reshape

@eqx.filter_jit
def _jit_compute_predictions(F_stack, W_stack):
    """Multiplie les features par les poids pour faire les prédictions (Batching)."""
    return F_stack @ W_stack


class Reservoir:
    '''
    Docstring to do.
    '''

    def __init__(self, W:jnp.array, chip:jpc_chip, mode='quadratures'):
        self.W = W
        self.chip = chip
        self.F = []
        self.mode = mode # si quadratures ou probas

    def push(self, data:jnp.array, quantum_parameters_list: list[QuantumParameters], 
             simulation_constants: JPCConfig, encoding_observable):
        '''
        Docstring to do.
        '''
        List_state_history = self.chip.run_simulation(data, quantum_parameters_list, encoding_observable=encoding_observable)
        match self.mode:
            case 'quadratures':
                F = [ build_f_quadratures(List_state_history[i], simulation_constants) for i in range(len(List_state_history)) ]
            case 'probas':
                F = [ build_f_probas(List_state_history[i], simulation_constants) for i in range(len(List_state_history)) ]
        self.F = F
        return self.F

    
    def train(self, data: jnp.array, label: jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        
        label_reshape = jnp.reshape(jnp.array(label), (len(label), 1))
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        
        # Empilement
        F_stack = jnp.stack(self.F) 
        
        # Appel de la fonction compilée !
        W_stack = _jit_compute_weights(F_stack, label_reshape)
        
        self.W = list(W_stack)
        return self.W
    
    def test(self, data: jnp.array, label_data: jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        
        # Empilement
        F_stack = jnp.stack(self.F)
        W_stack = jnp.stack(self.W)

        # Appel de la fonction compilée !
        Y_batch = _jit_compute_predictions(F_stack, W_stack)
        
        return list(Y_batch)


    '''
    def train(self, data:jnp.array, label:jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        
        Docstring.
        
        label_reshape = jnp.reshape(jnp.array(label), (len(label), 1))
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        
        # 1. On empile la liste en un seul gros Tenseur 3D
        F_stack = jnp.stack(self.F) 
        # 2. Le GPU calcule TOUTES les pseudo-inverses en même temps !
        F_inv = jnp.linalg.pinv(F_stack) 
        # 3. Multiplication matricielle en batch
        W_stack = F_inv @ label_reshape
        self.W = list(W_stack) # On remet en liste si ton code test() l'attend ainsi
        return self.W
    
    def test(self, data:jnp.array, label_data:jnp.array, 
              quantum_parameters_list: list[QuantumParameters], simulation_constants: JPCConfig, encoding_observable):
        Docstring to do.
        self.push(data, quantum_parameters_list, simulation_constants, encoding_observable)
        # 2. Vectorisation : On empile les listes en tenseurs 3D
        # F_batch shape: (Nb_Simulations, Temps, Features)
        # W_batch shape: (Nb_Simulations, Features, 1)
        F_batch = jnp.stack(self.F)
        W_batch = jnp.stack(self.W)

        # 3. Batch Matrix Multiplication (BMM)
        # L'opérateur @ de JAX détecte automatiquement le batching.
        # Le calcul (B, T, N) @ (B, N, 1) -> (B, T, 1) est fait en parallèle sur le GPU.
        Y_batch = F_batch @ W_batch 
        
        # 4. On retourne une liste pour garder la compatibilité avec ton interface
        return list(Y_batch)
    '''



@eqx.filter_jit
def _jit_format_quadratures(L_Ia, L_Qa, L_Ib, L_Qb, input_dim, nb_quadratures):
    """Fonction pure compilée par JAX pour redimensionner et empiler les tableaux."""
    L_Ia = L_Ia.reshape(-1, input_dim // nb_quadratures)
    L_Qa = L_Qa.reshape(-1, input_dim // nb_quadratures)
    L_Ib = L_Ib.reshape(-1, input_dim // nb_quadratures)
    L_Qb = L_Qb.reshape(-1, input_dim // nb_quadratures)
    return jnp.hstack((L_Ia, L_Qa, L_Ib, L_Qb))

# 2. TA FONCTION CLASSIQUE (Qui extrait les données et appelle la fonction JIT)
def build_f_quadratures(state_history: StateHistory, simulation_constants: JPCConfig) -> Array:
    nb_quadratures = 4
    input_dim = nb_quadratures * simulation_constants.MEASURE_RESOLUTION
    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION

    # Extraction (Python s'en charge très vite)
    L_Ia = state_history.quadratures.L_Ia[::step]
    L_Qa = state_history.quadratures.L_Qa[::step]
    L_Ib = state_history.quadratures.L_Ib[::step]
    L_Qb = state_history.quadratures.L_Qb[::step]

    # Calcul (Le GPU prend le relais grâce au JIT)
    F = _jit_format_quadratures(L_Ia, L_Qa, L_Ib, L_Qb, input_dim, nb_quadratures)
    return F      

@eqx.filter_jit
def _jit_format_probas(probas_exp: jnp.array, step: int, measure_resolution: int, nb_neurones: int):
    """
    Fonction pure JAX : Sous-échantillonne et redimensionne la matrice pour le multiplexage temporel.
    """
    # 1. Sous-échantillonnage temporel
    # probas_exp passe de (Temps_total_simu, nb_neurones) à (n * measure_resolution, nb_neurones)
    P_downsampled = probas_exp[::step, :]
    
    # 2. Multiplexage temporel (Chunking)
    # On regroupe les instants de mesure d'une même période sur une seule ligne.
    # On passe à (n, measure_resolution * nb_neurones)
    return P_downsampled.reshape(-1, measure_resolution * nb_neurones)

def build_f_probas(state_history, simulation_constants):
    """
    Construit la Feature Matrix F à partir des probabilités des niveaux de Fock.
    Chaque ligne correspond à UNE donnée de test (n), contenant l'évolution 
    des probabilités sur toute la période de mesure.
    """
    # 1. Calcul du pas d'échantillonnage
    step = simulation_constants.SIMULATION_RESOLUTION // simulation_constants.MEASURE_RESOLUTION
    
    # 2. Récupération des probabilités (Temps_total_simu, nb_neurones)
    probas_exp = state_history.photon_distribution.probas_exp
    
    # 3. On extrait le nombre de neurones dynamiquement (le nombre de colonnes)
    nb_neurones = probas_exp.shape[1]
    
    # 4. Application via le GPU
    F = _jit_format_probas(probas_exp, step, simulation_constants.MEASURE_RESOLUTION, nb_neurones)
    
    return F
        

'''
def build_f_quadratures(state_history: StateHistory, simulation_constants: JPCConfig) -> Array:
    
    Construit la feature matrix

    Returns
    -------
    F : Array
        Feature matrix F(X)
    
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
'''