import jax.numpy as jnp

from quantum_learn.build_f import BuildF
from quantum_simulation.history import StateHistory


class Reservoir:
    def __init__(self, build_f: BuildF) -> None:
        self.W: jnp.ndarray = jnp.array([])
        self.build_f: BuildF = build_f

    def __call__(self, state_history: StateHistory) -> jnp.ndarray:
        return self.forward(state_history)

    def fit(self, state_history: StateHistory, Y: jnp.ndarray) -> None:
        F = self.build_f(state_history)
        self.W = self.compute_weights(F, Y)

    def forward(self, state_history: StateHistory) -> jnp.ndarray:
        F = self.build_f(state_history)
        Y = self.compute_predictions(F)
        return Y

    @staticmethod
    def compute_weights(F: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """Calcule toutes les pseudo-inverses et les poids d'un coup (Batching)."""
        F_inv = jnp.linalg.pinv(F)
        return F_inv @ Y

    def compute_predictions(self, F: jnp.ndarray) -> jnp.ndarray:
        """Multiplie les features par les poids pour faire les prédictions (Batching)."""
        return F @ self.W
