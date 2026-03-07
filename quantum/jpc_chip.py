import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from qutip.ui import progress_bars


class JpcChip:
    PI = jnp.pi

    DIM_A = 10
    DIM_B = 12

    KAPPA_A = 2
    KAPPA_B = 2

    OMEGA_A = 9 * 1e3
    OMEGA_B = 1e4

    K_AA = 0.1
    K_BB = 0.1
    K_AB = 0.05

    EPSILON_A = 400
    EPSILON_B = 400

    a = dq.destroy(DIM_A)
    a_dag = a.dag()
    N_a = a_dag @ a

    b = dq.destroy(DIM_B)
    b_dag = b.dag()
    N_b = b_dag @ b

    H_kerr_a = K_AA * N_a @ N_a
    H_kerr_b = K_BB * N_b @ N_b
    H_cross = - K_AB * dq.tensor(N_a, N_b)
    H_kerr = dq.tensor(H_kerr_a, dq.eye(DIM_B)) + dq.tensor(dq.eye(DIM_A), H_kerr_b) + H_cross

    # drives
    H_d_a = 1j * jnp.sqrt(KAPPA_A) * (EPSILON_A.conjugate() * a - EPSILON_A * a_dag)
    H_d_b = 1j * jnp.sqrt(KAPPA_B) * (EPSILON_B.conjugate() * b - EPSILON_B * b_dag)
    H_d = dq.tensor(H_d_a, dq.eye(DIM_B)) + dq.tensor(dq.eye(DIM_A), H_d_b)


    PSI0 = [dq.tensor(dq.basis(DIM_A, 0), dq.basis(DIM_B, 0))] * 3

    jump_ops = [jnp.sqrt(KAPPA_A) * dq.tensor(a, dq.eye(DIM_B)) + jnp.sqrt(KAPPA_B) * dq.tensor(dq.eye(DIM_A), b)]  # Opérateurs de dissipation

    exp_ops = [dq.tensor(a, dq.eye(DIM_B)), dq.tensor(dq.eye(DIM_A), b)]  # Valeurs moyennes à calculer

    INCREMENT_TIME = 0.05  # Durée d'un train (1/8 de période)
    STEP_RESOLUTION = 50  # résolution pour dq

    def H0(self, g_conv, g_sq):
        """
        Renvoi l'Hamiltonien sans drive
        """
        return self.H_kerr + g_conv * (dq.tensor(self.a, self.b_dag) + dq.tensor(self.a_dag, self.b)) + g_sq * (
                dq.tensor(self.a, self.b) + dq.tensor(self.a_dag, self.b_dag))

    def get_next_state(self, x: float, psi, t: np.ndarray, params_G: np.ndarray):
        """
        Entrainement sur un train (1/8 de période)
        """
        Hd_a = 1j * jnp.sqrt(self.KAPPA_A) * (self.EPSILON_A.conjugate() * self.a - self.EPSILON_A * self.a_dag)
        Hd_b = 1j * jnp.sqrt(self.KAPPA_B) * (self.EPSILON_B.conjugate() * self.b - self.EPSILON_B * self.b_dag)
        Hd = dq.tensor(Hd_a, dq.eye(self.DIM_B)) + dq.tensor(dq.eye(self.DIM_A), Hd_b)
        H = [self.H0(g_conv, g_sq) + Hd * x for g_conv, g_sq in params_G]

        result = dq.mesolve(H, self.jump_ops, psi, t, exp_ops=self.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=False))

        return result

    def run_simulation(self, X: np.ndarray, params_G: np.ndarray) -> np.ndarray:

        """
        Entrainement sur toutes les données

        Entrées :
                données d'entrainement X = (list) vecteur de taille 8 x n_periodes
                params_G = liste de 3 couples [(g1, g2), (g1 + dg1, g2), (g1, g2 + dg2)]
                Nt = nombre de points / instants par train ### optionnel

        Sorties : features = Matrices F(X) de taille 64 x n_periodes

        """
        # First train
        time_interval = np.linspace(0, self.INCREMENT_TIME, self.STEP_RESOLUTION)

        nb_simus = len(params_G)
        nb_points = len(X)
        nb_points_per_period = 8
        nb_periods = len(X) // nb_points_per_period

        Quadratures = [Quadrature(nb_points, nb_periods, nb_points_per_period) for _ in range(nb_simus)]

        # Tableaux des features (sorties de la puce) -> Matrice de taille 64 x n_periodes

        psi = [dq.tensor(dq.fock(self.DIM_A, 0), dq.fock(self.DIM_B, 0))] * nb_simus


        for time in range(len(X)):

            result = self.get_next_state(X[time], psi, time_interval, params_G)
            time_interval += self.INCREMENT_TIME
            psi = [result.states[i][-1] for i in range(nb_simus)]

            for i, Q in enumerate(Quadratures):
                Q.update(result.expects[i], time)

        return np.stack([Q.build_F() for Q in Quadratures], axis=0)


class Quadrature:
    def __init__(self, nb_points: int, nb_periods: int, nb_points_per_period: int):
        self.nb_points: int = nb_points
        self.nb_periods: int = nb_periods
        self.nb_points_per_period: int = nb_points_per_period

        self.L_Ia: np.ndarray = np.zeros(nb_points)
        self.L_Qa: np.ndarray = np.zeros(nb_points)
        self.L_Ib: np.ndarray = np.zeros(nb_points)
        self.L_Qb: np.ndarray = np.zeros(nb_points)

    def update(self, expect, index: int) -> None:
        a_dq = expect[1]
        b_dq = expect[2]

        I_a = a_dq.real[-1]
        Q_a = a_dq.imag[-1]
        I_b = b_dq.real[-1]
        Q_b = b_dq.imag[-1]


        self.L_Ia[index] = I_a
        self.L_Qa[index] = Q_a
        self.L_Ib[index] = I_b
        self.L_Qb[index] = Q_b

    def build_F(self) -> np.ndarray:
        L_Ia = np.reshape(self.L_Ia, (self.nb_points_per_period, self.nb_periods))
        L_Qa = np.reshape(self.L_Qa, (self.nb_points_per_period, self.nb_periods))
        L_Ib = np.reshape(self.L_Ib, (self.nb_points_per_period, self.nb_periods))
        L_Qb = np.reshape(self.L_Qb, (self.nb_points_per_period, self.nb_periods))

        return np.vstack((
            np.vstack((L_Ia, L_Qa, L_Ib, L_Qb)),
             np.hstack((
                 np.zeros((4 * self.nb_points_per_period, 1)),
                 np.vstack((L_Ia[:, :-1], L_Qa[:, :-1], L_Ib[:, :-1], L_Qb[:, :-1]))
             ))
        ))