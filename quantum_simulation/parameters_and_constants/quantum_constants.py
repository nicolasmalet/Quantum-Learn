from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp


@dataclass
class QuantumConstants:
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

    EPSILON_A: float
    EPSILON_B: float
    DRIVE_DURATION: float

    def __post_init__(self):
        self.a = dq.destroy(self.DIM_A)
        self.a_dag = self.a.dag()
        self.N_a = self.a_dag @ self.a
        self.b = dq.destroy(self.DIM_B)
        self.b_dag = self.b.dag()
        self.N_b = self.b_dag @ self.b

        self.H_kerr_a = self.K_AA * self.N_a @ self.N_a
        self.H_kerr_b = self.K_BB * self.N_b @ self.N_b
        self.H_cross = - self.K_AB * dq.tensor(self.N_a, self.N_b)
        self.H_kerr = dq.tensor(self.H_kerr_a, dq.eye(self.DIM_B)) + dq.tensor(dq.eye(self.DIM_A),
                                                                               self.H_kerr_b) + self.H_cross

        self.H_da = dq.tensor(
            1j * jnp.sqrt(self.KAPPA_A) * (self.EPSILON_A.conjugate() * self.a - self.EPSILON_A * self.a_dag),
            dq.eye(self.DIM_B))
        self.H_db = dq.tensor(dq.eye(self.DIM_A), 1j * jnp.sqrt(self.KAPPA_B) * (
                self.EPSILON_B.conjugate() * self.b - self.EPSILON_B * self.b_dag))
        self.Hd = self.H_da + self.H_db

        self.vacuum_state = dq.tensor(dq.basis(self.DIM_A, 0),
                                      dq.basis(self.DIM_B, 0))  # états initiaux === vaccum states
        self.jump_ops = [jnp.sqrt(self.KAPPA_A) * dq.tensor(self.a, dq.eye(self.DIM_B)),
                         jnp.sqrt(self.KAPPA_B) * dq.tensor(dq.eye(self.DIM_A), self.b)]  # Opérateurs de dissipation
        self.exp_ops = [dq.tensor(self.a, dq.eye(self.DIM_B)),
                        dq.tensor(dq.eye(self.DIM_A), self.b)]  # Valeurs moyennes à calculer

    def H0(self, g_conv: float, g_sq: float):
        """
        Build the free-drive Hamiltonian.

        Parameters
        ----------
        g_conv : float
            conversion JRM mode coupling coefficient
        g_sq : float
            two mode squeezing JRM mode coupling coefficient

        Returns
        -------
        dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray (Dynamiqs Hamiltonian)
            Free-drive hamiltonian = Kerr effet + JRM contributions (conversion AND two mode squeezing)
        """
        return (self.H_kerr + g_conv * (dq.tensor(self.a, self.b_dag) + dq.tensor(self.a_dag, self.b))
                + g_sq * (dq.tensor(self.a, self.b) + dq.tensor(self.a_dag, self.b_dag)))
