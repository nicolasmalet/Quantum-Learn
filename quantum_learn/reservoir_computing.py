import jax.numpy as jnp
import numpy as np

from quantum_learn.build_f import BuildF
from quantum_simulation.history import StateHistory
from .types import Array

class Reservoir:
    def __init__(self, build_f: BuildF) -> None:
        self.W: Array = np.array([])
        self.build_f: BuildF = build_f

    def __call__(self, state_history: StateHistory) -> Array:
        return self.forward(state_history)

    def fit(self, state_history: StateHistory, Y: Array) -> None:
        F = self.build_f(state_history)
        self.W = self.compute_weights(F, Y)

    def forward(self, state_history: StateHistory) -> Array:
        F = self.build_f(state_history)
        Y = self.compute_predictions(F)
        return Y

    @staticmethod
    def compute_weights(F: Array, Y: Array) -> Array:
        """Calcule toutes les pseudo-inverses et les poids d'un coup (Batching)."""
        F_inv = jnp.linalg.pinv(F)
        return F_inv @ Y

    def compute_predictions(self, F: Array) -> Array:
        """Multiplie les features par les poids pour faire les prédictions (Batching)."""
        return F @ self.W
