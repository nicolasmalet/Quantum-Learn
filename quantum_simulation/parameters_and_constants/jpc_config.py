from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp
from zeroth.abstract import Summary

from .quantum_parameters import QuantumParameters


@dataclass
class JPCConfig(Summary):
    """
    Parameters
    ----------
    OMEGA_A : float
        Resonance frequency of mode a (GHz).
    OMEGA_B : float
        Resonance frequency of mode b (GHz).
    DIM_A : int
        Hilbert space truncation dimension for mode a.
    DIM_B : int
        Hilbert space truncation dimension for mode a.
    KAPPA_A : float
        leakage coefficient for resonator 1
    KAPPA_B : float
        leakage coefficient for resonator 2
    K_AA : float
        self Kerr coefficient for resonator 1
    K_BB : float
        self Kerr coefficient for resonator 2
    K_AB : float
        crossed Kerr coefficient for between resonators 1 and 2
    EPSILON_A : float
        drive a amplitude
    EPSILON_B : float
        drive b amplitude
    MEASURE_RESOLUTION: int
    SIMULATION_RESOLUTION: int

    Attributes
    ----------
    self.H0 :
        Builds the free-drive hamiltonian.

    """
    DIM_A: int
    DIM_B: int
    OMEGA_A: float
    OMEGA_B: float
    KAPPA_A: float
    KAPPA_B: float
    K_AA: float
    K_BB: float
    K_AB: float

    DRIVE_DURATION: float

    MEASURE_RESOLUTION: int
    SIMULATION_RESOLUTION: int

    def __post_init__(self):
        self.a = dq.destroy(self.DIM_A)
        self.a_dag = self.a.dag()
        self.N_a = self.a_dag @ self.a
        self.b = dq.destroy(self.DIM_B)
        self.b_dag = self.b.dag()
        self.N_b = self.b_dag @ self.b
        self.Ha = dq.tensor(self.N_a, dq.eye(self.DIM_B))
        self.Hb = dq.tensor(dq.eye(self.DIM_A), self.N_b)

        # Hamiltoniens effet Kerr
        self.H_kerr_a = self.K_AA * self.N_a @ self.N_a
        self.H_kerr_b = self.K_BB * self.N_b @ self.N_b
        self.H_cross = - self.K_AB * dq.tensor(self.N_a, self.N_b)
        self.H_kerr = dq.tensor(self.H_kerr_a, dq.eye(self.DIM_B)) + dq.tensor(dq.eye(self.DIM_A),
                                                                               self.H_kerr_b) + self.H_cross

        # Hamiltoniens Drive
        self.H_da = dq.tensor(
            jnp.sqrt(self.KAPPA_A) * (self.a + self.a_dag),
            dq.eye(self.DIM_B))
        self.H_db = dq.tensor(dq.eye(self.DIM_A), jnp.sqrt(self.KAPPA_B) * (
                self.b + self.b_dag))
        


        self.vacuum_state = dq.tensor(dq.basis(self.DIM_A, 0),
                                      dq.basis(self.DIM_B, 0))  # états initiaux === vaccum states
        self.jump_ops = [jnp.sqrt(self.KAPPA_A) * dq.tensor(self.a, dq.eye(self.DIM_B)),
                         jnp.sqrt(self.KAPPA_B) * dq.tensor(dq.eye(self.DIM_A), self.b)]  # Opérateurs de dissipation
        self.basic_exp_ops = [dq.tensor(self.a, dq.eye(self.DIM_B)),
                        dq.tensor(dq.eye(self.DIM_A), self.b), 
                        dq.tensor(self.N_a, dq.eye(self.DIM_B)),
                        dq.tensor(dq.eye(self.DIM_A), self.N_b)]  # Valeurs moyennes à calculer

        self.fock_ops = self.get_all_fock_projectors()

        self.exp_ops = self.basic_exp_ops + self.fock_ops



    def get_all_fock_projectors(self):
        """Génère la liste complète des projecteurs |n_a, n_b><n_a, n_b|."""
        ops = []
        for i in range(self.DIM_A):
            for j in range(self.DIM_B):
                # On construit le ket |i, j>
                ket = dq.tensor(dq.basis(self.DIM_A, i), dq.basis(self.DIM_B, j))
                # On ajoute le projecteur |i, j><i, j|
                ops.append(ket @ ket.dag())
        return ops

    def H_delta(self, delta_a, delta_b):
        return -delta_a * dq.tensor(self.N_a, dq.eye(self.DIM_B)) - delta_b * dq.tensor(dq.eye(self.DIM_A), self.N_b)
    
    def H_drive(self, epsilon_a, epsilon_b):
        H_da = dq.tensor(
            jnp.sqrt(self.KAPPA_A) * (epsilon_a.conjugate() * self.a + epsilon_a * self.a_dag),
            dq.eye(self.DIM_B))
        H_db = dq.tensor(dq.eye(self.DIM_A), jnp.sqrt(self.KAPPA_B) * (
                epsilon_b.conjugate() * self.b + epsilon_b * self.b_dag))
        return H_da + H_db
    
    def H_conv(self, g_conv):
        return g_conv.conjugate() * dq.tensor(self.a, self.b_dag) + g_conv * dq.tensor(self.a_dag, self.b)
    
    def H_sq(self, g_sq):
        return g_sq.conjugate() * dq.tensor(self.a, self.b) + g_sq * dq.tensor(self.a_dag, self.b_dag)
    
    
    def Build_H(self, quantum_parameters: QuantumParameters, encoding_observable='epsilon'):
        """
        Build the free-drive Hamiltonian.

        Parameters
        ----------
        quantum_parameters: QuantumParameters

        Returns
        -------
        dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray (Dynamiqs Hamiltonian)
            Free-drive hamiltonian = Kerr effet + JRM contributions (conversion AND two mode squeezing)
        """
        g_conv = quantum_parameters.g_conv
        g_sq = quantum_parameters.g_sq
        epsilon_a = quantum_parameters.epsilon_a
        epsilon_b = quantum_parameters.epsilon_b
        delta_a = quantum_parameters.delta_a
        delta_b = quantum_parameters.delta_b

        match encoding_observable:
            case 'epsilon':
                H = self.H_delta(delta_a, delta_b) + self.H_conv(g_conv) + self.H_sq(g_sq)
            case 'g_conv':
                H = self.H_delta(delta_a, delta_b) + self.H_drive(epsilon_a, epsilon_b) + self.H_sq(g_sq)
            case 'g_sq':
                H = self.H_delta(delta_a, delta_b) + self.H_drive(epsilon_a, epsilon_b) + self.H_conv(g_conv)

        return H
    
    def Encode_Data(self, data:jnp.array, O:float):
        O_encoded = data * O
        return O_encoded
    
    def Build_H_batched(self, quantum_parameters_list, encoding_observable='epsilon'):
        """
        Construit l'Hamiltonien pour tout un batch instantanément via broadcasting.
        """
        # 1. On extrait les paramètres dans des arrays JAX
        # Le [:, None, None] transforme un vecteur de taille (B,) en (B, 1, 1).
        # Cela permet à JAX de multiplier ce vecteur par des matrices (N, N).
        g_conv = jnp.array([qp.g_conv for qp in quantum_parameters_list])[:, None, None]
        g_sq = jnp.array([qp.g_sq for qp in quantum_parameters_list])[:, None, None]
        delta_a = jnp.array([qp.delta_a for qp in quantum_parameters_list])[:, None, None]
        delta_b = jnp.array([qp.delta_b for qp in quantum_parameters_list])[:, None, None]
        epsilon_a = jnp.array([qp.epsilon_a for qp in quantum_parameters_list])[:, None, None]
        epsilon_b = jnp.array([qp.epsilon_b for qp in quantum_parameters_list])[:, None, None]

        # 2. Appel direct ! JAX fait le produit sur tout le batch d'un coup.
        # Pas de boucle for, pas de création d'objets inutiles.
        match encoding_observable:
            case 'epsilon':
                H_batch = self.H_delta(delta_a, delta_b) + self.H_conv(g_conv) + self.H_sq(g_sq)
            case 'g_conv':
                H_batch = self.H_delta(delta_a, delta_b) + self.H_drive(epsilon_a, epsilon_b) + self.H_sq(g_sq)
            case 'g_sq':
                H_batch = self.H_delta(delta_a, delta_b) + self.H_drive(epsilon_a, epsilon_b) + self.H_conv(g_conv)

        # On fait un seul appel à .to_jax() pour tout le tenseur (B, N, N)
        return H_batch
    
    def H(self, quantum_parameters, data: jnp.array, time_interval: jnp.array, f_encoding,
          encoding_type='amplitude', encoding_observable='epsilon'):
        '''Docstring to do.'''


        # 1. Construction du batch H_free ultra-rapide
        H_free_batch = self.Build_H_batched(quantum_parameters, encoding_observable=encoding_observable)
        
        # 2. Conversion pour Dynamiqs
        H_statique = dq.constant(H_free_batch)



        match encoding_observable:
            case 'epsilon':
                epsilon_a = quantum_parameters[0].epsilon_a
                epsilon_b = quantum_parameters[0].epsilon_b
                values_a = f_encoding(epsilon_a, data)
                values_b = f_encoding(epsilon_b, data)
                H_encoded = (dq.pwc(time_interval, values_a, self.KAPPA_A * dq.tensor(self.a, dq.eye(self.DIM_B)))
                     + dq.pwc(time_interval, jnp.conj(values_a), self.KAPPA_A * dq.tensor(self.a_dag, dq.eye(self.DIM_B)))
                     + dq.pwc(time_interval, values_b, self.KAPPA_B * dq.tensor(dq.eye(self.DIM_A), self.b))
                     + dq.pwc(time_interval, jnp.conj(values_b), self.KAPPA_B * dq.tensor(dq.eye(self.DIM_A), self.b_dag))
                     )
            case 'g_conv':
                g_conv = quantum_parameters[0].g_conv
                values = f_encoding(g_conv, data)
                H_encoded = (dq.pwc(time_interval, values, dq.tensor(self.a_dag, self.b)) + 
                            dq.pwc(time_interval, jnp.conj(values), dq.tensor(self.a, self.b_dag)))
            case 'g_sq':
                g_sq = quantum_parameters[0].g_sq
                values = f_encoding(g_sq, data)
                H_encoded = (dq.pwc(time_interval, values, dq.tensor(self.a_dag, self.b_dag)) + 
                            dq.pwc(time_interval, jnp.conj(values), dq.tensor(self.a, self.b)))
                
        #H = H_free + H_encoded

        H_total = H_statique + H_encoded

        return H_total






