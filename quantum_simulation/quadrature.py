from .simulation_params import SimulationParams

import matplotlib.pyplot as plt
import numpy as np


class Quadrature:

    """
    Quadratures -> feature matrix

    This class implements a la structure de données qui stocke les quadratures
    du champ réfléchi au fil de l'évolution de l'état de la puce sous l'effet
    des drives (entraînement ou test)

    Parameters
    ----------
    params: SimulationParams
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

    def __init__(self, params: SimulationParams, expects):
        self.params = params
        step = params.SIMULATION_RESOLUTION // params.MEASURE_RESOLUTION

        self.L_Ia = expects[0].real[::step]
        self.L_Qa = expects[0].imag[::step]
        self.L_Ib = expects[1].real[::step]
        self.L_Qb = expects[1].imag[::step]

    def build_F(self, nb_periods_per_batch: int, nb_points_per_period: int) -> np.ndarray:
        """
        Construit la feature matrix selon la bonne notation

        Returns
        -------
        F : np.ndarray
            Feature matrix F(X)
        """

        L_Ia = self.L_Ia.reshape(nb_periods_per_batch, nb_points_per_period * self.params.MEASURE_RESOLUTION).T
        L_Qa = self.L_Qa.reshape(nb_periods_per_batch, nb_points_per_period * self.params.MEASURE_RESOLUTION).T
        L_Ib = self.L_Ib.reshape(nb_periods_per_batch, nb_points_per_period * self.params.MEASURE_RESOLUTION).T
        L_Qb = self.L_Qb.reshape(nb_periods_per_batch, nb_points_per_period * self.params.MEASURE_RESOLUTION).T
        return np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))


    def plot(self):
        # Plot les quadratures en fonction du temps

        X = np.arange(len(self.L_Ia))
        plt.plot(X, self.L_Ia, label="Ia")
        plt.plot(X, self.L_Qa, label="Qa")
        plt.plot(X, self.L_Ib, label="Ib")
        plt.plot(X, self.L_Qb, label="Qb")
        plt.legend()
        plt.show()
