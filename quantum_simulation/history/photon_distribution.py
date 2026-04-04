import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from ..parameters_and_constants import JPCConfig


class PhotonDistribution:
    """
    Docstring to do.
    """

    def __init__(self, jpc_config: JPCConfig, states, time_interval):
        self.jpc_config: JPCConfig = jpc_config
        self.DIM_A = jpc_config.DIM_A
        self.DIM_B = jpc_config.DIM_B
        self.Na = dq.expect(dq.tensor(jpc_config.N_a, dq.eye(self.DIM_B)), states)
        self.Nb = dq.expect(dq.tensor(dq.eye(self.DIM_A), jpc_config.N_b), states)
        self.Delta_Na = dq.expect(dq.tensor(jpc_config.N_a @ jpc_config.N_a, dq.eye(self.DIM_B)),
                                  states) - self.Na ** 2
        self.Delta_Nb = dq.expect(dq.tensor(dq.eye(self.DIM_A), jpc_config.N_b @ jpc_config.N_b),
                                  states) - self.Nb ** 2
        self.states = states
        self.time_interval = time_interval

        diag = jnp.diagonal(self.states.to_jax(), axis1=-2, axis2=-1)
        self.joint_proba = diag.reshape(states.shape[0], self.DIM_A, self.DIM_B).real
        self.P_a = self.joint_proba.sum(axis=2)
        self.P_b = self.joint_proba.sum(axis=1)

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

    def plot_joint_proba(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.subplots_adjust(bottom=0.2)

        # --- Normalisation log ---
        vmax = self.joint_proba.max()
        vmin = max(self.joint_proba[self.joint_proba > 0].min(), vmax * 1e-6)  # évite log(0)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        # --- Plot initial (k=0) ---
        data = np.array(self.joint_proba[0].T)
        data = np.clip(data, vmin, None)  # remplace les 0 par vmin

        im = ax.imshow(
            data,
            origin="lower",
            aspect="equal",
            cmap="inferno",
            norm=norm
        )
        plt.colorbar(im, ax=ax, label="Probabilité (log)")

        ax.set_xlabel(r"$i$ — mode $a$, état $|i\rangle$")
        ax.set_ylabel(r"$j$ — mode $b$, état $|j\rangle$")
        ax.set_xticks(range(self.DIM_A))
        ax.set_yticks(range(self.DIM_B))
        ax.set_xticklabels([f"$|{i}\\rangle$" for i in range(self.DIM_A)])
        ax.set_yticklabels([f"$|{j}\\rangle$" for j in range(self.DIM_B)])
        title = ax.set_title(f"$P(|i\\rangle|j\\rangle)$ — $t = {self.time_interval[0]:.4f}$")

        # --- Slider ---
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
        slider = Slider(ax=ax_slider, label="Temps $k$",
                        valmin=0, valmax=len(self.time_interval) - 1,
                        valinit=0, valstep=1, color="steelblue")

        # --- Callback ---
        def update(val):
            k = int(slider.val)
            data = np.clip(np.array(self.joint_proba[k].T), vmin, None)
            im.set_data(data)
            title.set_text(f"$P(|i\\rangle|j\\rangle)$ — $t = {self.time_interval[k]:.4f}$")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
        return fig, slider

    def plot_marginal_proba(self, mode="a"):
        """
        P     : shape (T, n)  — P_a ou P_b
        tsave : shape (T,)
        mode  : "a" ou "b" — pour les labels
        """
        P = self.P_a if mode == "a" else self.P_b
        T, n = P.shape
        label_i = r"$i$" if mode == "a" else r"$j$"
        label_P = rf"$P_{mode}({label_i})$"
        state_label = lambda k: rf"$|{k}\rangle$"

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.subplots_adjust(bottom=0.25)

        # --- Plot initial ---
        x = np.arange(n)
        bars = ax.bar(x, P[0], color="steelblue", edgecolor="white")

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, P.max() * 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels([state_label(k) for k in x])
        ax.set_xlabel(rf"État de Fock du mode ${mode}$")
        ax.set_ylabel(label_P)
        title = ax.set_title(rf"{label_P} — $t = {self.time_interval[0]:.4f}$")

        # --- Slider ---
        ax_slider = plt.axes((0.15, 0.08, 0.7, 0.04))
        slider = Slider(
            ax=ax_slider,
            label="Temps $k$",
            valmin=0,
            valmax=T - 1,
            valinit=0,
            valstep=1,
            color="steelblue"
        )

        # --- Callback ---
        def update(val):
            k = int(slider.val)
            for bar, h in zip(bars, P[k]):
                bar.set_height(h)
            title.set_text(rf"{label_P} — $t = {self.time_interval[k]:.4f}$")
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        return fig, slider
