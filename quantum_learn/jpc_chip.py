import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from qutip.ui import progress_bars
import matplotlib.pyplot as plt


class Quadrature:

    """
    Quadratures -> feature matrix

    This class implements a la structure de données qui stocke les quadratures 
    du champ refléchi au fil de l'évolution de l'état de la puce sous l'effet 
    des drives (entraînement ou test) 

    Parameters
    ----------
    nb_points : int
        Nombre de points.
    nb_periods : int
        nombre de periodes.
    nb_points_per_period : int
        nombre de points par période.
    nb_points_per_drive : int
        nombre de points par drive


    Attributes
    ----------
    update : 
        Update les quadratures.
    build_F : ndarray
        Build la feature matrix F(X)

    Notes
    -----
    RAS au max

    References
    ----------
    Je l'ai vu dans un rêve...
    """

    def __init__(self, nb_points: int, nb_periods: int, nb_points_per_period: int, nb_points_per_drive = 1):
        self.nb_points            = nb_points
        self.nb_periods           = nb_periods
        self.nb_points_per_period = nb_points_per_period
        self.nb_points_per_drive = nb_points_per_drive

        self.L_Ia = np.zeros(nb_points * nb_points_per_drive)
        self.L_Qa = np.zeros(nb_points * nb_points_per_drive)
        self.L_Ib = np.zeros(nb_points * nb_points_per_drive)
        self.L_Qb = np.zeros(nb_points * nb_points_per_drive)

    def update(self, expect, index: int, multiple_inputs=True) -> None:
        """
        Update les quadratures

        Parameters
        ----------
        expect : list
            liste contenant <a> et <b>
        idex : int
            indice de l'update
        multiple_inputs : boolean
            indique si les quadratures a et b sont différentes
        """

        ### 50 k tq t[50k] = 5k ns pour k\in{0,..,9}
        a_dq = expect[0]
        
        for k in range(self.nb_points_per_drive):
            self.L_Ia[self.nb_points_per_drive * index + k] = a_dq[50 * k].real
            self.L_Qa[self.nb_points_per_drive * index + k] = a_dq[50 * k].imag

        if multiple_inputs:
            b_dq = expect[1]
            for k in range(self.nb_points_per_drive):
                self.L_Ib[self.nb_points_per_drive * index + k] = b_dq[50 * k].real
                self.L_Qb[self.nb_points_per_drive * index + k] = b_dq[50 * k].imag
        
        

    def build_F(self, multiple_inputs=True) -> np.ndarray:
        """
        Construit la feature matrix selon la bonne notation

        Parameters
        ----------
        multiple_inputs : boolean
            indique si les quadratures a et b sont différentes

        Returns
        -------
        F : jnp.array 
            Feature matrix F(X)
        """

        L_Ia = self.L_Ia.reshape(self.nb_periods, self.nb_points_per_period * self.nb_points_per_drive).T
        L_Qa = self.L_Qa.reshape(self.nb_periods, self.nb_points_per_period * self.nb_points_per_drive).T
        #print("Ia =", self.L_Ia, "   taille =", self.L_Ia.shape) 
        #print("Qa =", self.L_Qa, "   taille =", self.L_Qa.shape) 
        #print("Iamod =", L_Ia, "   taille =", L_Ia.shape) 
        #print("Qamod =", L_Qa, "   taille =", L_Qa.shape) 

        if multiple_inputs :
            L_Ib = self.L_Ib.reshape(self.nb_periods, self.nb_points_per_period * self.nb_points_per_drive).T
            L_Qb = self.L_Qb.reshape(self.nb_periods, self.nb_points_per_period * self.nb_points_per_drive).T
            bloc_A = np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))
            return np.array(bloc_A)

        '''
        if multiple_inputs :
            bloc_B = np.hstack((
                jnp.zeros((4 * self.nb_points_per_period, 1)),
                jnp.vstack((L_Ia[:, :-1], L_Qa[:, :-1],
                        L_Ib[:, :-1], L_Qb[:, :-1]))
            ))
            return jnp.vstack((bloc_A, bloc_B))
        '''
    
        bloc_A = np.vstack((L_Ia, L_Qa))
        return np.array(bloc_A)

    def plot(self, multiple_inputs=True):
        '''
        Plot les quadratures en fonction du temps

        Parameters
        ----------
        multiple_inputs : boolean
            indique si les quadratures a et b sont différentes
        '''

        X = range(len(self.L_Ia))
        plt.plot(X, self.L_Ia, label="Ia")
        plt.plot(X, self.L_Qa, label="Qa")
        if multiple_inputs :
            plt.plot(X, self.L_Ib, label="Ib")
            plt.plot(X, self.L_Qb, label="Qb")
        plt.legend()
        plt.show()



class JpcChip:
    """
    Josephson Parametric Converter (JPC) made of two resonators chip whom contains
    one mode each, for neuromorphic quantum computing simulations.

    This class implements a truncated Hilbert-space model of a
    JPC chip and computes perturbative corrections to the
    effective Hamiltonian.

    Parameters
    ----------
    OMEGA_A : float
        Resonance frequency of mode a (GHz).
    OMEGA_B : float
        Resonance frequency of mode b (GHz).
    g : float
        Nonlinear coupling strength.
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
    EPSILON_A : float
        drive a amplitude
    EPSILON_B : float
        drive b amplitude


    Attributes
    ----------
    H0 : 
        Builds the free-drive hamiltonian.
    get_next_state : 
        Résout l'équation de Lindblad pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial.
    run_simulation : 
        Entraîne la puce sur toutes les données
        -> résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

    Notes
    -----
    Units:  ħ = 1.
            time in microseconds
            frequency in MHz
            drive amplitude in \sqrt{MHz}
    The model assumes zero temperature.

    References
    ----------
    Cohen-Tannoudji, Quantum Mechanics Vol. 2.
    """

    PI = jnp.pi
    ### Grandeurs physiques de la puce ###
    DIM_A = 10
    DIM_B = 10
    OMEGA_A = 1e4
    OMEGA_B = 9 * 1e3
    KAPPA_A = 17
    KAPPA_B = 21
    K_AA = 0.1
    K_BB = 0.1
    K_AB = 0.05

    ### Paramètres du drive ####
    EPSILON_A = 550  # amplitude drive a 
    EPSILON_B = 550  # amplitude drive b
    INCREMENT_TIME = 0.05  # Durée d'un drive


    ### Building dynamiqs operators ###
    a = dq.destroy(DIM_A)
    a_dag = a.dag()
    N_a = a_dag @ a
    b = dq.destroy(DIM_B)
    b_dag = b.dag()
    N_b = b_dag @ b

    ### Building Hamiltonians ###
    H_kerr_a = K_AA * N_a @ N_a
    H_kerr_b = K_BB * N_b @ N_b
    H_cross = -K_AB * dq.tensor(N_a, N_b)
    H_kerr = dq.tensor(H_kerr_a, dq.eye(DIM_B)) + dq.tensor(dq.eye(DIM_A), H_kerr_b) + H_cross
    H_da = dq.tensor(1j * jnp.sqrt(KAPPA_A) * (EPSILON_A.conjugate() * a - EPSILON_A * a_dag), dq.eye(DIM_B))
    H_db = dq.tensor(dq.eye(DIM_A), 1j * jnp.sqrt(KAPPA_B) * (EPSILON_B.conjugate() * b - EPSILON_B * b_dag))
    Hd = H_da + H_db

    ### Paramètres pour dq.mesolve ###
    VACCUM_STATE = dq.tensor(dq.basis(DIM_A, 0), dq.basis(DIM_B, 0)) # états initiaux === vaccum states 
    jump_ops = [jnp.sqrt(KAPPA_A) * dq.tensor(a, dq.eye(DIM_B)), jnp.sqrt(KAPPA_B) * dq.tensor(dq.eye(DIM_A), b)]  # Opérateurs de dissipation
    exp_ops = [dq.tensor(a, dq.eye(DIM_B)), dq.tensor(dq.eye(DIM_A), b)]  # Valeurs moyennes à calculer
    STEP_RESOLUTION = 500  # résolution des simulations Dynamiqs


    def H0(self, g_conv, g_sq):
        """
        Build the free-drive Hamiltonian.

        Parameters
        ----------
        g_conv : float
            conversion JRM mode coupling coefficient
        g_sq : float
            two mode squeezing JRM mode coupling coefficient

        Returns
        -------
        dynamiqs.qarrays.sparsedia_qarray.SparseDIAQArray (Dynamiqs Hamiltonian)
            Free-drive hamiltonian = Kerr effet + JRM contributions (conversion AND two mode squeezing)
        """
        return self.H_kerr + g_conv * (dq.tensor(self.a, self.b_dag) + dq.tensor(self.a_dag, self.b)) + g_sq * (
                dq.tensor(self.a, self.b) + dq.tensor(self.a_dag, self.b_dag))

    def get_next_state(self, x, psi, t: jnp.ndarray, params_G: list, multiple_inputs=True):
        """
        Résout l'équation de Lindblad pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

        Parameters
        ----------
        x : float or 
            entrée(s) encodée(s) en amplitude du drive
        psi : 
            état initial de la simulation à t=t[0]
        t : jnp.ndarray
            tableau des instants de la simulation
        params_G : list
            liste des valeurs du couple (g_conv, g_sq)
        multiple_inputs : boolean
            précise si il y a plusieurs données d'entrée point par point
        Returns
        -------
        dynamiqs.result.MESolveResult
            Résultat de la simulation dynamiqs
        """
        if multiple_inputs:
            xa, xb = x[0], x[1]
            H = [self.H0(g_conv, g_sq) + xa * self.H_da + xb * self.H_db for g_conv, g_sq in params_G]
        else :
            H = [self.H0(g_conv, g_sq) + self.Hd * x for g_conv, g_sq in params_G]

        result = dq.mesolve(H, self.jump_ops, psi, t, exp_ops=self.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=False))

        return result

    def run_simulation(self, X: jnp.ndarray, params_G: list, multiple_inputs=True, plot=False, params_feature=[8, 10]) -> jnp.ndarray:
        """
        Entraîne la puce sur toutes les données
        -> résout l'équation de Lindblad drive après drive pour plusieurs valeurs possibles du couple
        (g_conv, g_sq) sur les instants t avec psi comme état initial

        Parameters
        ----------
        X : jnp.ndarray
            données d'entraînement encodées en amplitude du drive
        params_G : list
            liste des valeurs du couple (g_conv, g_sq)
            cad [(g1, g2), (g1 + dg1, g2), (g1, g2 + dg2)]
        multiple_inputs : boolean
            précise si il y a plusieurs données d'entrée point par point
        params_feature : list (optional)
            précise "nb_points_per_period" et "nb_points_per_drive" pour la construction des quadratures

        Returns
        -------
        F1 : jnp.array of shape 64 x len(X)
            Feature matrix for the simulation 1
        F2 : jnp.array of shape 64 x len(X)
            Feature matrix for the simulation 2
        F3 : jnp.array of shape 64 x len(X)
            Feature matrix for the simulation 3
        """
        # First train
        time_interval = jnp.linspace(0, self.INCREMENT_TIME, self.STEP_RESOLUTION)
        psi = self.VACCUM_STATE

        nb_simus = len(params_G)
        nb_points = len(X)
        nb_points_per_period = params_feature[0]
        nb_points_per_drive = params_feature[1]
        nb_periods = len(X) // nb_points_per_period

        Quadratures = [Quadrature(nb_points, nb_periods, nb_points_per_period, nb_points_per_drive=nb_points_per_drive) for _ in range(nb_simus)]

        # Tableaux des features (sorties de la puce) -> Matrice de taille 64 x n_periodes
        psi = self.VACCUM_STATE

        for time in range(len(X)):

            result = self.get_next_state(X[time], psi, time_interval, params_G, multiple_inputs=multiple_inputs)
            time_interval += self.INCREMENT_TIME
            psi = [result.states[i][-1] for i in range(nb_simus)]
            # update des quadratures
            for i, Q in enumerate(Quadratures):
                Q.update(result.expects[i], time, multiple_inputs=multiple_inputs)

        if plot:
            Quadratures[0].plot(multiple_inputs=multiple_inputs)

        return jnp.stack([Q.build_F(multiple_inputs=multiple_inputs) for Q in Quadratures], axis=0)
