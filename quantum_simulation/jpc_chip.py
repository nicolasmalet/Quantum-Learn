from .quantum_params import QuantumParams
from .simulation_params import SimulationParams
from .quadrature import Quadrature

import jax.numpy as jnp
import dynamiqs as dq
import numpy as np

class JpcChip:
    """
    Josephson Parametric Converter (JPC) made of two resonators chip whom contains
    one mode each, for neuromorphic quantum computing simulations.

    This class implements a truncated Hilbert-space model of a
    JPC chip and computes perturbative corrections to the
    effective Hamiltonian.

    Attributes
    ----------
    self.run_simulation :
        Entraîne la puce sur toutes les données
        -> résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

    Notes
    -----
    Units:  ħ = 1.
            time in microseconds
            frequency in MHz
            drive amplitude in sqrt{MHz}
    The model assumes zero temperature.

    References
    ----------
    Cohen-Tannoudji, Quantum Mechanics Vol. 2.
    """  # résolution des simulations Dynamiqs

    def __init__(self, quantum_parameters: QuantumParams, simulation_params: SimulationParams):
        self.quantum_parameters = quantum_parameters
        self.simulation_params = simulation_params

    def run_simulation(self, X: np.ndarray, params_G: np.ndarray, plot: bool = False) -> list[Quadrature]:
        """
        Entraîne la puce sur toutes les données
        -> résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

        Parameters
        ----------
        X : jnp.ndarray
            données d'entraînement encodées en amplitude du drive
        params_G : np.ndarray
            array des valeurs du couple (g_conv, g_sq)
            i.e. [(g1, g2), (g1 + dg1, g2), (g1, g2 + dg2)]
        plot: bool (optional)
            si True, plot les quadratures
        Returns
        -------
        F1 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 1
        F2 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 2
        F3 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 3
        """

        nb_simulations = len(params_G)

        # Tableaux des features (sorties de la puce) -> Matrice de taille 64 x n_periodes

        time_interval = jnp.linspace(0, self.quantum_parameters.DRIVE_DURATION * len(X),
                                     self.simulation_params.SIMULATION_RESOLUTION * len(X))
        psi = self.quantum_parameters.vacuum_state
        tab_data = np.repeat(X, self.simulation_params.SIMULATION_RESOLUTION)[:-1]

        H_drive = dq.pwc(time_interval, tab_data, self.quantum_parameters.Hd)
        H0s = dq.stack([
            self.quantum_parameters.H0(g_conv, g_sq)
            for g_conv, g_sq in params_G
        ])
        H = H0s + H_drive

        result = dq.mesolve(H, self.quantum_parameters.jump_ops, psi, time_interval,
                            exp_ops=self.quantum_parameters.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=True, save_states=False))

        Quadratures = [Quadrature(self.simulation_params, result.expects[i]) for i in range(nb_simulations)]

        if plot:
            Quadratures[0].plot()

        return Quadratures
