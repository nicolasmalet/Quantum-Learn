import dynamiqs as dq
import jax.numpy as jnp

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

    def run_simulation(self, X: jnp.ndarray, quantum_parameters: list[QuantumParameters]) -> list[StateHistory]:
        """
        Résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

        Parameters
        ----------
        X : Array
            données d'entraînement encodées en amplitude du drive
        quantum_parameters : list[QuantumParameters]
        Returns
        -------
        list[StateHistory]
        """

        nb_simulations = len(quantum_parameters)
        time_interval = jnp.linspace(0, self.config.DRIVE_DURATION * X.size,
                                     self.config.SIMULATION_RESOLUTION * X.size)
        psi = self.config.vacuum_state
        data = jnp.repeat(X, self.config.SIMULATION_RESOLUTION)[:-1]

        H = self.config.H(quantum_parameters=quantum_parameters,
                          data=data,
                          time_interval=time_interval,
                          f_encoding=lambda _X, _data: _data * _X)

        result = dq.mesolve(H=H,
                            jump_ops=self.config.jump_ops,
                            rho0=psi,
                            tsave=time_interval,
                            exp_ops=self.config.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=True, save_states=True))

        return [StateHistory(self.config, result.expects[i], result.states[i],
                             time_interval) for i in range(nb_simulations)]
