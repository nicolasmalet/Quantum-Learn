from dataclasses import dataclass
from typing import Callable

import dynamiqs as dq
import jax.numpy as jnp
from zeroth.abstract import Summary

from .quantum_parameters import QuantumParameters


@dataclass
class JPCConfig(Summary):
    """
    Parameters
    ----------
    OMEGA_A : float
        Resonance frequency of mode a (GHz).
    OMEGA_B : float
        Resonance frequency of mode b (GHz).
    DIM_A : int
        Hilbert space truncation dimension for mode a.
    DIM_B : int
        Hilbert space truncation dimension for mode a.
    KAPPA_A : float
        leakage coefficient for resonator 1
    KAPPA_B : float
        leakage coefficient for resonator 2
    K_AA : float
        self Kerr coefficient for resonator 1
    K_BB : float
        self Kerr coefficient for resonator 2
    K_AB : float
        crossed Kerr coefficient for between resonators 1 and 2
    MEASURE_RESOLUTION: int
    SIMULATION_RESOLUTION: int

    Attributes
    ----------
    self.H0 :
        Builds the free-drive hamiltonian.

    """
    DIM_A: int
    DIM_B: int
    OMEGA_A: float
    OMEGA_B: float
    KAPPA_A: float
    KAPPA_B: float
    K_AA: float
    K_BB: float
    K_AB: float

    DRIVE_DURATION: float

    MEASURE_RESOLUTION: int
    SIMULATION_RESOLUTION: int



