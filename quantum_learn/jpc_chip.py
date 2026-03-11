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

    def __init__(self, nb_simus, nb_points, nb_periods, PER_PERIODS, MEASURE_RESOLUTION, SIMU_RESOLUTION, expects):
        self.nb_simus = nb_simus
        self.nb_points = nb_points
        self.nb_periods = nb_periods
        self.PER_PERIODS = PER_PERIODS
        self.MEASURE_RESOLUTION = MEASURE_RESOLUTION
        self.SIMU_RESOLUTION = SIMU_RESOLUTION
        step = SIMU_RESOLUTION // MEASURE_RESOLUTION

        print(expects[0])

        self.L_Ia = expects[0].real[::step] 
        self.L_Qa = expects[0].imag[::step] 
        self.L_Ib = expects[1].real[::step] 
        self.L_Qb = expects[1].imag[::step] 
        #print(expects[0].shape)
        #print(expects[0].shape)
        #print(self.L_Ia[0].shape)
        #print("Step =", step)





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
        
        L_Ia = self.L_Ia.reshape(self.nb_periods, self.PER_PERIODS * self.MEASURE_RESOLUTION).T
        L_Qa = self.L_Qa.reshape(self.nb_periods, self.PER_PERIODS * self.MEASURE_RESOLUTION).T
        L_Ib = self.L_Ib.reshape(self.nb_periods, self.PER_PERIODS * self.MEASURE_RESOLUTION).T
        L_Qb = self.L_Qb.reshape(self.nb_periods, self.PER_PERIODS * self.MEASURE_RESOLUTION).T
        return np.vstack((L_Ia, L_Qa, L_Ib, L_Qb))


    def plot(self):
        '''
        Plot les quadratures en fonction du temps

        Parameters
        ----------
        multiple_inputs : boolean
            indique si les quadratures a et b sont différentes
        '''

        X = range(self.nb_points * self.MEASURE_RESOLUTION)
        plt.plot(X, self.L_Ia, label="Ia")
        plt.plot(X, self.L_Qa, label="Qa")
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
    DIM_A = 15
    DIM_B = 15
    OMEGA_A = 1e4
    OMEGA_B = 9 * 1e3
    KAPPA_A = 17
    KAPPA_B = 21
    K_AA = 0.1
    K_BB = 0.1
    K_AB = 0.05

    ### Paramètres du drive ####
    EPSILON_A = 20  # amplitude drive a 
    EPSILON_B = 20 # amplitude drive b
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
    #H_da = dq.tensor(1j * jnp.sqrt(KAPPA_A) * (EPSILON_A.conjugate() * a - EPSILON_A * a_dag), dq.eye(DIM_B))
    #H_db = dq.tensor(dq.eye(DIM_A), 1j * jnp.sqrt(KAPPA_B) * (EPSILON_B.conjugate() * b - EPSILON_B * b_dag))
    H_da = dq.tensor( jnp.sqrt(KAPPA_A) * (EPSILON_A.conjugate() * a + EPSILON_A * a_dag), dq.eye(DIM_B))
    H_db = dq.tensor(dq.eye(DIM_A),jnp.sqrt(KAPPA_B) * (EPSILON_B.conjugate() * b + EPSILON_B * b_dag))
    Hd = H_da + H_db

    ### Paramètres pour dq.mesolve ###
    VACCUM_STATE = dq.tensor(dq.basis(DIM_A, 0), dq.basis(DIM_B, 0)) # états initiaux === vaccum states 
    jump_ops = [jnp.sqrt(KAPPA_A) * dq.tensor(a, dq.eye(DIM_B)), jnp.sqrt(KAPPA_B) * dq.tensor(dq.eye(DIM_A), b)]  # Opérateurs de dissipation
    exp_ops = [dq.tensor(a, dq.eye(DIM_B)), dq.tensor(dq.eye(DIM_A), b)]  # Valeurs moyennes à calculer

    # QUADRATURES FEATURES
    PER_PERIODS = 8
    MEASURE_RESOLUTION = 10
    SIMU_RESOLUTION = 200  # résolution des simulations Dynamiqs



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

    def Data_simu(self, t, data):
        '''
        Crée une liste X de même taille que le temps.
        '''
        tab_data = np.zeros(len(t))
        for i in range(len(data)):
            tab_data[i * self.MEASURE_RESOLUTION: (i+1) * self.MEASURE_RESOLUTION] = data[i]
        return tab_data[:-1]


    def run_simulation(self, X: jnp.ndarray, params_G: list, plot=False) -> jnp.ndarray:
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

        nb_simus = len(params_G)
        nb_points = len(X)
        nb_periods = len(X) // self.PER_PERIODS


        # Tableaux des features (sorties de la puce) -> Matrice de taille 64 x n_periodes
        time_interval = jnp.linspace(0, self.INCREMENT_TIME * len(X), self.SIMU_RESOLUTION * len(X))
        psi = self.VACCUM_STATE
        tab_data = self.Data_simu(time_interval, X)

        #H = [self.H0(g_conv, g_sq) + dq.pwc(time_interval, tab_data, self.Hd) for g_conv, g_sq in params_G]
        H = [self.H0(g_conv, g_sq) for g_conv, g_sq in params_G]

        result = dq.mesolve(H, self.jump_ops, psi, time_interval, exp_ops=self.exp_ops,
                            options=dq.Options(cartesian_batching=False, progress_meter=False))
        
        Quadratures = [Quadrature(nb_simus, nb_points, nb_periods, self.PER_PERIODS, self.MEASURE_RESOLUTION, self.SIMU_RESOLUTION, result.expects[i]) for i in range(nb_simus)]
      
        if plot:
            Quadratures[0].plot()

        return np.stack([Q.build_F() for Q in Quadratures], axis=0)
