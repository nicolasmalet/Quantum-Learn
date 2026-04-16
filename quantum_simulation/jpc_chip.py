import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from .encoding_functions import EncodingFunction, Linear
from .hamiltonian import Hamiltonian
from .history.state_history import StateHistory
from .parameters_and_constants.jpc_config import JPCConfig
from .parameters_and_constants.quantum_parameters import QuantumParameters


class JPCChip:
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

    def __init__(self, jpc_config: JPCConfig, encoding_parameters: tuple = ("gsq",),
                 encoding_function: EncodingFunction = Linear) -> None:
        self.config: JPCConfig = jpc_config
        self.hamiltonian = Hamiltonian(jpc_config, encoding_parameters, encoding_function)

    def run_simulation(self, X: np.ndarray, quantum_parameters_list: list[QuantumParameters]) -> list[StateHistory]:
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
        list[StateHistory]
        """

        X = jnp.asarray(X)

        nb_simulations = len(quantum_parameters_list)
        time_interval = jnp.linspace(0, self.config.DRIVE_DURATION * X.size,
                                     self.config.SIMULATION_RESOLUTION * X.size)
        psi = self.hamiltonian.vacuum_state
        tab_data = jnp.repeat(X, self.config.SIMULATION_RESOLUTION)[:-1]

        H = self.hamiltonian.H_tot(quantum_parameters_list=quantum_parameters_list,
                                   data=tab_data,
                                   time_interval=time_interval)

        result = dq.mesolve(H, self.hamiltonian.jump_ops, psi, time_interval,
                            exp_ops=self.hamiltonian.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=True,
                                               save_states=False))

        return [StateHistory(jpc_config=self.config,
                             expects=result.expects[i],
                             time_interval=time_interval) for i in range(nb_simulations)]
