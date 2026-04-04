import dynamiqs as dq
import matplotlib.pyplot as plt
import numpy as np

from .photon_distribution import PhotonDistribution
from .quadratures import Quadratures
from ..parameters_and_constants import JPCConfig


class StateHistory:
    """
    Docstring to do.
    """

    def __init__(self, jpc_config: JPCConfig, expects, states, time_interval):
        self.jpc_config: JPCConfig = jpc_config
        self.expects = expects
        self.states = states
        self.time_interval = time_interval
        self.photon_distribution = PhotonDistribution(jpc_config, states, time_interval)
        self.quadratures = Quadratures(expects)
        self.DIM_A = self.jpc_config.DIM_A
        self.DIM_B = self.jpc_config.DIM_B

    def plot_trace_verification(self):
        """
        Calcule et plot Tr([a, a†] rho(t)) et Tr([b, b†] rho(t)) en fonction du temps.
        Les deux doivent être égaux à 1 à tout instant (normalisation de rho).
        """
        # --- Calcul mode a ---
        comm_a = self.jpc_config.a @ self.jpc_config.a_dag - self.jpc_config.a_dag @ self.jpc_config.a
        comm_a_full = dq.tensor(comm_a, dq.eye(self.DIM_B))
        trace_a = np.array(dq.expect(comm_a_full, self.states).real)

        # --- Calcul mode b ---
        comm_b = self.jpc_config.b @ self.jpc_config.b_dag - self.jpc_config.b_dag @ self.jpc_config.b
        comm_b_full = dq.tensor(dq.eye(self.DIM_A), comm_b)
        trace_b = np.array(dq.expect(comm_b_full, self.states).real)

        tsave_np = np.array(self.time_interval)
        ecart_a = np.abs(trace_a - 1.0).max()
        ecart_b = np.abs(trace_b - 1.0).max()
        valid_a = ecart_a < 5 * 1e-2
        valid_b = ecart_b < 5 * 1e-2

        # --- Plot ---
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        for ax, trace, ecart, valid, mode in zip(
                [ax_a, ax_b],
                [trace_a, trace_b],
                [ecart_a, ecart_b],
                [valid_a, valid_b],
                ["a", "b"]
        ):
            comm_label = rf"$\mathrm{{Tr}}([{mode},{mode}^\dagger]\rho(t))$"

            ax.plot(tsave_np, trace, color="steelblue", lw=1.5, label=comm_label)
            ax.axhline(1.0, color="green", lw=1, ls="--", label="Référence = 1")
            ax.axhline(1.05, color="red", lw=1, ls=":", label="Seuil ±5%")
            ax.axhline(0.95, color="red", lw=1, ls=":")
            ax.fill_between(tsave_np, 0.95, 1.05, color="green", alpha=0.1)

            ax.set_ylabel(comm_label)
            ax.set_title(
                rf"Mode ${mode}$ — Ecart max = {ecart:.2e} — " +
                ("Valide à 1%" if valid else "Invalide à 5%")
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        ax_b.set_xlabel("Temps")
        fig.suptitle(r"Vérification de la normalisation de $\rho(t)$", fontsize=13)
        plt.tight_layout()
        plt.show()

        return trace_a, trace_b
