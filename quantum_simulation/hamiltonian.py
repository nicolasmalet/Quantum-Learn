import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs.time_qarray import PWCTimeQArray

from quantum_simulation.parameters_and_constants.jpc_config import JPCConfig
from quantum_simulation.parameters_and_constants.quantum_parameters import QuantumParameters


class Hamiltonian:
    def __init__(self, jpc_config: JPCConfig):
        self.jpc_config: JPCConfig = jpc_config

        self.a = dq.destroy(self.jpc_config.DIM_A)
        self.a_dag = self.a.dag()
        self.N_a = self.a_dag @ self.a
        self.b = dq.destroy(self.jpc_config.DIM_B)
        self.b_dag = self.b.dag()
        self.N_b = self.b_dag @ self.b
        self.Ha = dq.tensor(self.N_a, dq.eye(self.jpc_config.DIM_B))
        self.Hb = dq.tensor(dq.eye(self.jpc_config.DIM_A), self.N_b)

        # Hamiltoniens effet Kerr
        self.H_kerr_a = self.jpc_config.K_AA * self.N_a @ self.N_a
        self.H_kerr_b = self.jpc_config.K_BB * self.N_b @ self.N_b
        self.H_cross = - self.jpc_config.K_AB * dq.tensor(self.N_a, self.N_b)
        self.H_kerr = dq.tensor(self.H_kerr_a, dq.eye(self.jpc_config.DIM_B)) + dq.tensor(dq.eye(self.jpc_config.DIM_A),
                                                                               self.H_kerr_b) + self.H_cross

        # Hamiltoniens Drive
        self.H_da = dq.tensor(
            jnp.sqrt(self.jpc_config.KAPPA_A) * (self.a + self.a_dag),
            dq.eye(self.jpc_config.DIM_B))
        self.H_db = dq.tensor(dq.eye(self.jpc_config.DIM_A), jnp.sqrt(self.jpc_config.KAPPA_B) * (
                self.b + self.b_dag))

        self.vacuum_state = dq.tensor(dq.basis(self.jpc_config.DIM_A, 0),
                                      dq.basis(self.jpc_config.DIM_B, 0))  # états initiaux === vaccum states
        self.jump_ops = [jnp.sqrt(self.jpc_config.KAPPA_A) * dq.tensor(self.a, dq.eye(self.jpc_config.DIM_B)),
                         jnp.sqrt(self.jpc_config.KAPPA_B) * dq.tensor(dq.eye(self.jpc_config.DIM_A),
                                                                       self.b)]  # Opérateurs de dissipation
        self.basic_exp_ops = [dq.tensor(self.a, dq.eye(self.jpc_config.DIM_B)),
                              dq.tensor(dq.eye(self.jpc_config.DIM_A), self.b),
                              dq.tensor(self.N_a, dq.eye(self.jpc_config.DIM_B)),
                              dq.tensor(dq.eye(self.jpc_config.DIM_A), self.N_b)]  # Valeurs moyennes à calculer

        self.fock_ops = self.get_all_fock_projectors()

        self.exp_ops = self.basic_exp_ops + self.fock_ops

    def get_all_fock_projectors(self) -> list[dq.QArray]:
        """Génère la liste complète des projecteurs |n_a, n_b><n_a, n_b|."""
        ops = []
        for i in range(self.jpc_config.DIM_A):
            for j in range(self.jpc_config.DIM_B):
                # On construit le ket |i, j>
                ket = dq.tensor(dq.basis(self.jpc_config.DIM_A, i), dq.basis(self.jpc_config.DIM_B, j))
                # On ajoute le projecteur |i, j><i, j|
                ops.append(ket @ ket.dag())
        return ops

    def H_delta(self, delta_a: jnp.ndarray, delta_b: jnp.ndarray, time_interval: jnp.ndarray) -> PWCTimeQArray:
        return (dq.pwc(time_interval, delta_a,
                       - dq.tensor(self.N_a, dq.eye(self.jpc_config.DIM_B))) +
                dq.pwc(time_interval, delta_b,
                       - dq.tensor(dq.eye(self.jpc_config.DIM_A), self.N_b)))

    def H_drive(self, epsilon_a: jnp.ndarray, epsilon_b: jnp.ndarray, time_interval: jnp.ndarray) -> PWCTimeQArray:
        # Constantes multiplicatives pour les opérateurs de drive
        kappa_a_sqrt = jnp.sqrt(self.jpc_config.KAPPA_A)
        kappa_b_sqrt = jnp.sqrt(self.jpc_config.KAPPA_B)

        # Application de dq.pwc pour chaque terme (conjugué et normal)
        return (dq.pwc(time_interval, epsilon_a.conjugate(),
                       kappa_a_sqrt * dq.tensor(self.a, dq.eye(self.jpc_config.DIM_B))) +
                dq.pwc(time_interval, epsilon_a,
                       kappa_a_sqrt * dq.tensor(self.a_dag, dq.eye(self.jpc_config.DIM_B))) +
                dq.pwc(time_interval, epsilon_b.conjugate(),
                       kappa_b_sqrt * dq.tensor(dq.eye(self.jpc_config.DIM_A), self.b)) +
                dq.pwc(time_interval, epsilon_b,
                       kappa_b_sqrt * dq.tensor(dq.eye(self.jpc_config.DIM_A), self.b_dag)))

    def H_conv(self, g_conv: jnp.ndarray, time_interval: jnp.ndarray) -> PWCTimeQArray:
        return (dq.pwc(time_interval, g_conv.conjugate(),
                       dq.tensor(self.a, self.b_dag)) +
                dq.pwc(time_interval, g_conv,
                       dq.tensor(self.a_dag, self.b)))

    def H_sq(self, g_sq: jnp.ndarray, time_interval: jnp.ndarray) -> PWCTimeQArray:
        return (dq.pwc(time_interval, g_sq.conjugate(),
                       dq.tensor(self.a, self.b)) +
                dq.pwc(time_interval, g_sq,
                       dq.tensor(self.a_dag, self.b_dag)))


    def H_tot(self, quantum_parameters_list: list[QuantumParameters], data: jnp.ndarray, time_interval: jnp.ndarray):
        """
        Assemble un Hamiltonien batché pour N simulations en parallèle.
        """
        delta_a_vals = jnp.stack([p.get_value("delta_a", data) for p in quantum_parameters_list])
        delta_b_vals = jnp.stack([p.get_value("delta_b", data) for p in quantum_parameters_list])
        epsilon_a_vals = jnp.stack([p.get_value("epsilon_a", data) for p in quantum_parameters_list])
        epsilon_b_vals = jnp.stack([p.get_value("epsilon_b", data) for p in quantum_parameters_list])
        g_conv_vals = jnp.stack([p.get_value("g_conv", data) for p in quantum_parameters_list])
        g_sq_vals = jnp.stack([p.get_value("g_sq", data) for p in quantum_parameters_list])

        H_static = self.H_kerr_a + self.H_kerr_b + self.H_cross

        H_batched = (
            H_static +
            self.H_delta(delta_a_vals, delta_b_vals, time_interval) +
            self.H_drive(epsilon_a_vals, epsilon_b_vals, time_interval) +
            self.H_conv(g_conv_vals, time_interval) +
            self.H_sq(g_sq_vals, time_interval)
        )

        return H_batched