import dynamiqs as dq
import jax.numpy as jnp

from .hamiltonian import Hamiltonian
from .history.state_history import StateHistory
from .parameters_and_constants.jpc_config import JPCConfig
from .parameters_and_constants.quantum_parameters import QuantumParameters


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

    def __init__(self, jpc_config: JPCConfig):
        self.config: JPCConfig = jpc_config
        self.hamiltonian = Hamiltonian(jpc_config)

    def run_simulation(self, X: jnp.ndarray, quantum_parameters_list: list[QuantumParameters]) -> list[StateHistory]:
        """
        Entraîne la puce sur toutes les données
        -> résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

        Parameters
        ----------
        X : Array
            données d'entraînement encodées en amplitude du drive
        quantum_parameters_list : list[QuantumParameters]
        Returns
        -------
        F1 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 1
        F2 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 2
        F3 : np.array of shape 64 x len(X)
            Feature matrix for the simulation 3
        """

        nb_simulations = len(quantum_parameters_list)
        time_interval = jnp.linspace(0, self.config.DRIVE_DURATION * X.size,
                                     self.config.SIMULATION_RESOLUTION * X.size)
        psi = self.hamiltonian.vacuum_state
        tab_data = jnp.repeat(X, self.config.SIMULATION_RESOLUTION)[:-1]

        H = self.hamiltonian.H_tot(quantum_parameters_list=quantum_parameters_list,
                                   data=tab_data,
                                   time_interval=time_interval)

        methode_integration = dq.method.Dopri5(max_steps=1_000_000,
                                               atol=1e-8,
                                               rtol=1e-8)

        result = dq.mesolve(H, self.hamiltonian.jump_ops, psi, time_interval,
                            exp_ops=self.hamiltonian.exp_ops,
                            method=methode_integration,
                            options=dq.Options(cartesian_batching=False, progress_meter=True,
                                               save_states=False))

        return [StateHistory(jpc_config=self.config,
                             expects=result.expects[i],
                             time_interval=time_interval) for i in range(nb_simulations)]
