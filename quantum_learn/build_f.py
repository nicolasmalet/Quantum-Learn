from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import JPCConfig


@dataclass(frozen=True)
class BuildFConfig(ABC):
    @abstractmethod
    def instantiate(self, jpc_config: JPCConfig) -> BuildF:
        ...


class BuildF(ABC):
    output_dim: int

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, state_history: StateHistory) -> np.ndarray:
        ...


@dataclass(frozen=True)
class BuildFQuadraturesConfig(BuildFConfig):
    def instantiate(self, jpc_config: JPCConfig) -> BuildF:
        return BuildFQuadratures(jpc_config)


class BuildFQuadratures(BuildF):
    def __init__(self, jpc_config):
        self.jpc_config: JPCConfig = jpc_config

        self.step = self.jpc_config.SIMULATION_RESOLUTION // self.jpc_config.MEASURE_RESOLUTION

        self.nb_quadratures = 4
        self.output_dim = self.nb_quadratures * self.jpc_config.MEASURE_RESOLUTION

    def __call__(self, state_history: StateHistory) -> np.ndarray:
        """
        Construit la feature matrix

        Returns
        -------
        F : jnp.ndarray
            Feature matrix F(X)
        """

        L_Ia = state_history.quadratures.L_Ia[::self.step]
        L_Qa = state_history.quadratures.L_Qa[::self.step]
        L_Ib = state_history.quadratures.L_Ib[::self.step]
        L_Qb = state_history.quadratures.L_Qb[::self.step]

        L_Ia = L_Ia.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Qa = L_Qa.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Ib = L_Ib.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Qb = L_Qb.reshape(-1, self.output_dim // self.nb_quadratures)

        F = np.hstack((L_Ia, L_Qa, L_Ib, L_Qb))

        return F


@dataclass(frozen=True)
class BuildFQuadraturesPolynomialsConfig(BuildFConfig):
    def instantiate(self, jpc_config: JPCConfig) -> BuildF:
        return BuildFQuadratures(jpc_config)


class BuildFQuadraturesPolynomials(BuildF):
    def __init__(self, jpc_config: JPCConfig):
        self.jpc_config: JPCConfig = jpc_config

        self.step = self.jpc_config.SIMULATION_RESOLUTION // self.jpc_config.MEASURE_RESOLUTION

        self.nb_quadratures = 14
        self.output_dim = self.nb_quadratures * self.jpc_config.MEASURE_RESOLUTION

    def __call__(self, state_history: StateHistory) -> np.ndarray:
        """
        Construit la feature matrix

        Returns
        -------
        F : jnp.ndarray
            Feature matrix F(X)
        """
        L_Ia = state_history.quadratures.L_Ia[::self.step]
        L_Qa = state_history.quadratures.L_Qa[::self.step]
        L_Ib = state_history.quadratures.L_Ib[::self.step]
        L_Qb = state_history.quadratures.L_Qb[::self.step]

        L_Ia = L_Ia.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Qa = L_Qa.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Ib = L_Ib.reshape(-1, self.output_dim // self.nb_quadratures)
        L_Qb = L_Qb.reshape(-1, self.output_dim // self.nb_quadratures)

        F = np.hstack((L_Ia, L_Qa, L_Ib, L_Qb,
                       L_Ia ** 2, L_Qa ** 2, L_Ib ** 2, L_Qb ** 2,
                       L_Ia * L_Qa, L_Ia * L_Ib, L_Ia * L_Qb, L_Qa * L_Ib, L_Qa * L_Qb, L_Ib * L_Qb))

        return F


@dataclass(frozen=True)
class BuildFPhotonDistributionConfig(BuildFConfig):
    def instantiate(self, jpc_config: JPCConfig) -> BuildF:
        return BuildFQuadratures(jpc_config)


class BuildFPhotonDistribution(BuildF):
    def __init__(self, jpc_config: JPCConfig, clip_probas: int):
        self.jpc_config: JPCConfig = jpc_config

        self.step = self.jpc_config.SIMULATION_RESOLUTION // self.jpc_config.MEASURE_RESOLUTION
        self.clip_probas = clip_probas
        self.output_dim = clip_probas ** 2 * self.jpc_config.MEASURE_RESOLUTION

    def __call__(self, state_history: StateHistory) -> np.ndarray:
        measures = state_history.photon_distribution.joint_proba[::self.step, :self.clip_probas, :self.clip_probas]
        F = measures.reshape(-1, self.output_dim)
        return F
