import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from .hamiltonian import Hamiltonian
from .history.simulation_result import SimulationResult
from .parameters_and_constants import JPCConfig, QuantumParameters, Encoding


class Simulator:
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

    def __init__(self, jpc_config: JPCConfig, encoding: Encoding) -> None:
        self.config: JPCConfig = jpc_config
        self.encoding: Encoding = encoding
        self.hamiltonian = Hamiltonian(jpc_config, encoding)

    def __call__(self, X: np.ndarray, quantum_parameters_list: list[QuantumParameters]) -> list[SimulationResult]:
        return self.run_simulation(X, quantum_parameters_list)

    def run_simulation(self, X: np.ndarray, quantum_parameters_list: list[QuantumParameters]) -> list[SimulationResult]:
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
        list[SimulationResult]
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

        solver_config = dq.method.Dopri5(max_steps=10**9)

        result = dq.mesolve(H=H,
                            jump_ops=self.hamiltonian.jump_ops,
                            rho0=psi,
                            tsave=time_interval,
                            exp_ops=self.hamiltonian.exp_ops,
                            method=solver_config,
                            options=dq.Options(cartesian_batching=False,
                                               progress_meter=True,
                                               save_states=False))

        return [SimulationResult(jpc_config=self.config,
                                 expects=result.expects[i],
                                 time_interval=time_interval) for i in range(nb_simulations)]
