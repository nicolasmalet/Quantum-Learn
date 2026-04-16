import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class PhotonDistribution:
    def __init__(self, jpc_config, expects, time_interval):
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
        self.joint_proba = np.asarray(joint_proba_jax)
        self.P_a = np.asarray(P_a_jax)
        self.P_b = np.asarray(P_b_jax)

        # --- CORRECTION 2 : Protection mémoire du slider ---
        self.current_slider = None

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
            data_to_plot = np.log10(self.joint_proba + eps)  # np au lieu de jnp
            vmin, vmax = -6, 0
            cbar_label = 'Log10(Probabilité)'
            title_prefix = "Proba Jointe (Log10)"
        else:
            data_to_plot = self.joint_proba
            vmin, vmax = 0, np.max(self.joint_proba)
            cbar_label = 'Probabilité'
            title_prefix = "Proba Jointe"

        im = ax.imshow(data_to_plot[0], origin='lower', aspect='auto',
                       extent=(-0.5, self.DIM_B - 0.5, -0.5, self.DIM_A - 0.5),
                       cmap='viridis', vmin=vmin, vmax=vmax)

        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel('Photons mode B')
        ax.set_ylabel('Photons mode A')
        title = ax.set_title(f"{title_prefix} — t = {self.time_interval[0]:.2f}")

        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))

        # On sauvegarde le slider dans 'self' pour éviter sa destruction
        self.current_slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval) - 1, valinit=0, valfmt='%d')

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
            P_plot = np.log10(P_raw + eps)  # np au lieu de jnp
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

        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))

        # On sauvegarde le slider dans 'self' pour éviter sa destruction
        self.current_slider = Slider(ax_slider, 'Temps', 0, len(self.time_interval) - 1, valinit=0, valfmt='%d')

        def update(val):
            idx = int(self.current_slider.val)
            for i, b in enumerate(bars):
                b.set_height(P_plot[idx, i])
            title.set_text(f"{title_prefix} — t = {self.time_interval[idx]:.2f}")
            fig.canvas.draw_idle()

        self.current_slider.on_changed(update)
        plt.show()
