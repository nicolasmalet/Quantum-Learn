import matplotlib.pyplot as plt
import numpy as np

from ..parameters_and_constants.simulation_constants import SimulationConstants


class Quadratures:
    """
    Quadratures -> feature matrix

    This class implements a la structure de données qui stocke les quadratures
    du champ réfléchi au fil de l'évolution de l'état de la puce sous l'effet
    des drives (entraînement ou test)

    Parameters
    ----------
    params: SimulationConstants
    expects

    Attributes
    ----------

    self.build_F : ndarray
        Build la feature matrix F(X)

    Notes
    -----
    RAS au max

    References
    ----------
    Je l'ai vu dans un rêve...
    """

    def __init__(self, expects):
        self.L_Ia = expects[0].real
        self.L_Qa = expects[0].imag
        self.L_Ib = expects[1].real
        self.L_Qb = expects[1].imag

    def plot(self) -> None:
        # Plot les quadratures en fonction du temps

        X = np.arange(len(self.L_Ia))
        plt.plot(X, self.L_Ia, label="Ia")
        plt.plot(X, self.L_Qa, label="Qa")
        plt.plot(X, self.L_Ib, label="Ib")
        plt.plot(X, self.L_Qb, label="Qb")
        plt.legend()
        plt.show()
