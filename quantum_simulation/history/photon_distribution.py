
'''
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from ..parameters_and_constants import JPCConfig
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import jax.numpy as jnp

class PhotonDistribution:
    def __init__(self, jpc_config, expects, time_interval, nb_neurones=9):
        self.Na = expects[2]
        self.Nb = expects[3]
        self.jpc_config = jpc_config
        self.DIM_A = jpc_config.DIM_A
        self.DIM_B = jpc_config.DIM_B
        self.time_interval = time_interval
        
        # 1. Calculs rapides sur GPU avec JAX
        probs_reelles = jnp.real(jnp.array(expects[4:]))
        all_fock_probs = jnp.clip(probs_reelles, a_min=0.0, a_max=1.0)
        
        joint_proba_jax = all_fock_probs.T.reshape(
            len(time_interval), self.DIM_A, self.DIM_B
        )

        P_a_jax = joint_proba_jax.sum(axis=2)
        P_b_jax = joint_proba_jax.sum(axis=1)

        # --- CORRECTION 1 : Conversion en Numpy ---
        # Indispensable pour que Matplotlib puisse animer sans geler
        self.joint_proba = np.array(joint_proba_jax)
        self.P_a = np.array(P_a_jax)
        self.P_b = np.array(P_b_jax)

        # --- CORRECTION 2 : Protection mémoire du slider ---
        self.current_slider = None 

        # --- NOUVEAUTÉ : Extraction de probas_exp ---
        # On calcule la taille de la sous-grille (ex: 9 neurones -> N_max = 3)
        N_max = int(np.sqrt(nb_neurones))
        if N_max * N_max != nb_neurones:
            raise ValueError("nb_neurones doit être un carré parfait (ex: 4, 9, 16) pour inclure toutes les combinaisons croisées.")
        # On découpe la grille (Temps, N_max, N_max) et on l'aplatit en (Temps, nb_neurones)
        # On garde précieusement ce tableau en JAX pour que l'entraînement (Matrice W)
        # reste 100% sur le GPU sans transfert vers le CPU !
        self.probas_exp = joint_proba_jax[:, :N_max, :N_max].reshape(len(time_interval), nb_neurones)



    def Plot_Mean_Photon_Number(self):
        """
        Docstring to do
        """

        plt.figure()
        plt.plot(self.time_interval, self.Na, label=r"$<N_a>$")
        plt.plot(self.time_interval, self.Nb, label=r"$<N_b>$")
        plt.grid()
        plt.legend()
        plt.show()

        return self.Na, self.Nb

    def plot_joint_proba(self, use_log=False):
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.25)

        if use_log:
            eps = 1e-10
            data_to_plot = np.log10(self.joint_proba + eps) # np au lieu de jnp
            vmin, vmax = -6, 0
            cbar_label = 'Log10(Probabilité)'
            title_prefix = "Proba Jointe (Log10)"
        else:
            data_to_plot = self.joint_proba
            vmin, vmax = 0, np.max(self.joint_proba)
            cbar_label = 'Probabilité'
            title_prefix = "Proba Jointe"

        im = ax.imshow(data_to_plot[0], origin='lower', aspect='auto', 
                       extent=[-0.5, self.DIM_B-0.5, -0.5, self.DIM_A-0.5],
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel('Photons mode B')
        ax.set_ylabel('Photons mode A')
        title = ax.set_title(f"{title_prefix} — t = {self.time_interval[0]:.2f}")

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        # On sauvegarde le slider dans 'self' pour éviter sa destruction
        self.current_slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval)-1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(self.current_slider.val)
            im.set_data(data_to_plot[idx])
            title.set_text(f"{title_prefix} — t = {self.time_interval[idx]:.2f}")
            fig.canvas.draw_idle()

        self.current_slider.on_changed(update)
        plt.show()

    def plot_marginal_proba(self, mode="a", use_log=False):
        P_raw = self.P_a if mode == "a" else self.P_b
        n_max = self.DIM_A if mode == "a" else self.DIM_B
        
        if use_log:
            eps = 1e-10
            P_plot = np.log10(P_raw + eps) # np au lieu de jnp
            y_min, y_max = -6, 0
            y_label = f"Log10 P(n_{mode})"
            title_prefix = f"Marginale mode {mode} (Log10)"
        else:
            P_plot = P_raw
            y_min, y_max = 0, 1.1
            y_label = f"Probabilité P(n_{mode})"
            title_prefix = f"Marginale mode {mode}"

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.subplots_adjust(bottom=0.25)

        x = np.arange(n_max)
        bars = ax.bar(x, P_plot[0], color="steelblue", edgecolor="white")

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(f"État de Fock |n_{mode}>")
        ax.set_ylabel(y_label)
        title = ax.set_title(f"{title_prefix} — t = {self.time_interval[0]:.2f}")

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        # On sauvegarde le slider dans 'self' pour éviter sa destruction
        self.current_slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval)-1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(self.current_slider.val)
            for i, b in enumerate(bars):
                b.set_height(P_plot[idx, i])
            title.set_text(f"{title_prefix} — t = {self.time_interval[idx]:.2f}")
            fig.canvas.draw_idle()

        self.current_slider.on_changed(update)
        plt.show()











'''
class PhotonDistribution:
    """
    Docstring to do.
    """

    def __init__(self, jpc_config: JPCConfig, expects, time_interval):
        self.jpc_config: JPCConfig = jpc_config
        self.DIM_A = jpc_config.DIM_A
        self.DIM_B = jpc_config.DIM_B
        self.Na = expects[2]
        self.Nb = expects[3]
        self.Delta_Na = 0
        self.Delta_Nb = 0
        #self.Delta_Na = dq.expect(dq.tensor(jpc_config.N_a @ jpc_config.N_a, dq.eye(self.DIM_B)),
        #                          states) - self.Na ** 2
        #self.Delta_Nb = dq.expect(dq.tensor(dq.eye(self.DIM_A), jpc_config.N_b @ jpc_config.N_b),
        #                          states) - self.Nb ** 2
        self.time_interval = time_interval


        # On suppose que les 4 premiers indices de expects sont <a>, <b>, <Na>, <Nb>
        # On extrait les probas brutes et on force en réels
        probs_brutes = jnp.real(jnp.array(expects[4:]))
        
        # CORRECTION : On coupe tout ce qui est en dessous de 0 et au-dessus de 1
        all_fock_probs = jnp.clip(probs_brutes, a_min=0.0, a_max=1.0)
        
        # Reconstruction de la probabilité jointe P(na, nb, t)
        # Forme attendue : (Temps, DIM_A, DIM_B)
        self.joint_proba = jnp.array(all_fock_probs).T.reshape(
            len(time_interval), self.DIM_A, self.DIM_B
        )

        # Calcul des probabilités marginales par sommation
        self.P_a = self.joint_proba.sum(axis=2) # Somme sur b -> P(a)
        self.P_b = self.joint_proba.sum(axis=1) # Somme sur a -> P(b)

    def Plot_Mean_Photon_Number(self):
        """
        Docstring to do
        """

        plt.figure()
        plt.plot(self.time_interval, self.Na, label=r"$<N_a>$")
        plt.plot(self.time_interval, self.Nb, label=r"$<N_b>$")
        plt.grid()
        plt.legend()
        plt.show()

        return self.Na, self.Nb

    def Plot_Sigma_Photon_Number(self):
        """
        Docstring to do
        """

        plt.figure()
        plt.plot(self.time_interval, jnp.sqrt(self.Delta_Na), label=r"$\Delta N_a$")
        plt.plot(self.time_interval, jnp.sqrt(self.Delta_Nb), label=r"$\Delta N_b$")
        plt.grid()
        plt.legend()
        plt.show()

        return self.Na, self.Nb
    

    def plot_joint_proba(self, use_log=False):
        """Affiche la probabilité jointe P(na, nb). Si use_log=True, affiche Log10(P)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.25)

        # Préparation des données (Linéaire ou Log)
        if use_log:
            eps = 1e-10  # Empêche l'erreur log(0)
            data_to_plot = jnp.log10(self.joint_proba + eps)
            vmin, vmax = -6, 0  # Échelle typique de 10^-6 à 10^0
            cbar_label = 'Log10(Probabilité)'
            title_prefix = "Proba Jointe (Log10)"
        else:
            data_to_plot = self.joint_proba
            vmin, vmax = 0, jnp.max(self.joint_proba)
            cbar_label = 'Probabilité'
            title_prefix = "Proba Jointe"

        # Affichage initial
        im = ax.imshow(data_to_plot[0], origin='lower', aspect='auto', 
                       extent=[-0.5, self.DIM_B-0.5, -0.5, self.DIM_A-0.5],
                       cmap='viridis', vmin=vmin, vmax=vmax)
        
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel('Photons mode B')
        ax.set_ylabel('Photons mode A')
        title = ax.set_title(f"{title_prefix} — t = {self.time_interval[0]:.2f}")

        # Slider temporel
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval)-1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(slider.val)
            im.set_data(data_to_plot[idx])
            title.set_text(f"{title_prefix} — t = {self.time_interval[idx]:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    def plot_marginal_proba(self, mode="a", use_log=False):
        """Affiche la marginale P(n). Si use_log=True, affiche Log10(P)."""
        P_raw = self.P_a if mode == "a" else self.P_b
        n_max = self.DIM_A if mode == "a" else self.DIM_B
        
        # Préparation des données et des axes
        if use_log:
            eps = 1e-10
            P_plot = jnp.log10(P_raw + eps)
            y_min, y_max = -6, 0
            y_label = f"Log10 P(n_{mode})"
            title_prefix = f"Marginale mode {mode} (Log10)"
        else:
            P_plot = P_raw
            y_min, y_max = 0, 1.1
            y_label = f"Probabilité P(n_{mode})"
            title_prefix = f"Marginale mode {mode}"

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.subplots_adjust(bottom=0.25)

        x = np.arange(n_max)
        bars = ax.bar(x, P_plot[0], color="steelblue", edgecolor="white")

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(f"État de Fock |n_{mode}>")
        ax.set_ylabel(y_label)
        title = ax.set_title(f"{title_prefix} — t = {self.time_interval[0]:.2f}")

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval)-1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(slider.val)
            for i, b in enumerate(bars):
                # En mode log, les valeurs nulles donnent ~ -10, on force la hauteur de la barre
                b.set_height(P_plot[idx, i]) 
            title.set_text(f"{title_prefix} — t = {self.time_interval[idx]:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
'''