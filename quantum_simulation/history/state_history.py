import matplotlib.pyplot as plt
import numpy as np

from .photon_distribution import PhotonDistribution
from .quadratures import Quadratures
from ..parameters_and_constants import JPCConfig


class StateHistory:
    """
    Docstring to do.
    """

    def __init__(self, jpc_config: JPCConfig, expects, time_interval):
        self.jpc_config: JPCConfig = jpc_config
        self.expects = expects
        self.time_interval = time_interval
        self.photon_distribution = PhotonDistribution(jpc_config, expects, time_interval)
        self.quadratures = Quadratures(expects)
        self.DIM_A = self.jpc_config.DIM_A
        self.DIM_B = self.jpc_config.DIM_B

    def plot_commutator_verification(self, threshold_pct=5.0):
        """
        Vérifie la validité de la troncature via les commutateurs.
        Affiche l'intervalle 1 +/- seuil% et le pourcentage de réussite.
        """
        p_last_a = self.photon_distribution.P_a[:, -1]
        p_last_b = self.photon_distribution.P_b[:, -1]

        # Formule du commutateur tronqué
        comm_a = 1 - self.jpc_config.DIM_A * p_last_a
        comm_b = 1 - self.jpc_config.DIM_B * p_last_b

        # Définition des bornes
        threshold = threshold_pct / 100.0
        lower, upper = 1.0 - threshold, 1.0 + threshold

        # Calcul du pourcentage de temps passé dans l'intervalle
        valid_a = np.mean((comm_a >= lower) & (comm_a <= upper)) * 100
        valid_b = np.mean((comm_b >= lower) & (comm_b <= upper)) * 100

        plt.figure(figsize=(10, 5))
        plt.plot(self.time_interval, comm_a, label=r"$\langle [a, a^\dagger] \rangle$")
        plt.plot(self.time_interval, comm_b, label=r"$\langle [b, b^\dagger] \rangle$")

        # Affichage de l'intervalle de confiance (zone grise)
        plt.fill_between(self.time_interval, lower, upper, color='gray', alpha=0.2,
                         label=f"Intervalle de confiance ({threshold_pct}%)")
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        # Informations au-dessus du graphe
        plt.title(f"Score de validité : Mode A ({valid_a:.1f}%) | Mode B ({valid_b:.1f}%)\n"
                  f"Vérification de la troncature (Seuil: {threshold_pct}%)")
        plt.xlabel("Temps (μs)")
        plt.ylabel("Valeur moyenne")
        plt.ylim(0.5, 1.1)  # Centré sur 1 avec de la place pour voir la chute
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_trace_integrity(self):
        """
        Vérifie que la somme des probabilités vaut 1.
        Affiche la moyenne et la variance au-dessus du graphe.
        """
        # La trace est la somme de la probabilité jointe P(na, nb)
        trace = self.photon_distribution.joint_proba.sum(axis=(1, 2))

        avg_trace = np.mean(trace)
        var_trace = np.var(trace)

        plt.figure(figsize=(10, 5))
        plt.plot(self.time_interval, trace, label=r"Tr($\rho$)", color="forestgreen")
        plt.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Centrage de l'affichage : on prend une marge de +/- 0.05 autour de la moyenne
        # ou au moins une zone couvrant le 1.0
        plt.ylim(min(avg_trace, 1.0) - 0.05, max(avg_trace, 1.0) + 0.05)

        # Affichage des statistiques
        plt.title(f"Moyenne Trace : {avg_trace:.6f} | Variance : {var_trace:.2e}\n"
                  f"Vérification de l'intégrité de la Trace")
        plt.xlabel("Temps (μs)")
        plt.ylabel(r"Tr($\rho$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
