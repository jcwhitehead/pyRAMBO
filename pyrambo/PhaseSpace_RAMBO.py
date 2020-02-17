from pyrambo.PhaseSpace import PhaseSpace
from pyrambo.FourVector import FourVec
from pyrambo.ParticleFourMom import ParticleFourMom
import numpy as np

#TODO: write tests.
#TODO: clean up code.
#TODO: documentation (docstrings -> .


class PhaseSpace_RAMBO(PhaseSpace):
    """Subclass of PhaseSpace for RAMBO-generated phase-space points."""

    def __init__(self, npar, totalmom, massarray=None):
        # change if migrating to Python 3 to super().__init__(p)
        if isinstance(totalmom, int) and np.isclose(totalmom, 0):
            totalmom = np.zeros(4)
        super().__init__(npar, totalmom, massarray)
        Vn = [None, None, 3.9788735772973836e-2,
              1.2598255637968553e-4,
              1.3296564302788846e-07,
              7.016789757994904e-11,
              2.2217170114046142e-14,
              4.689729110704849e-18,
              7.070965840156896e-22,
              7.995966835036095e-26,
              7.032646042454293e-30,
              4.948305702493552e-34,
              2.8486829022771386e-38]
        if not self.npar:
            self.npar = npar
        if npar < 12:
            self.vol = Vn[npar]
        else:
            self.vol = V_n(npar)

    # definition differs from GitHub implementation by a factor of (1/2pi)^4: theirs is (1/2pi)^(3n) instead.
    # Whizzard matches GitHub.
    # Definition here matches paper and Herwig.
    def V_n(n):
        V = 1 / (2 * np.pi) ** (3 * n - 4) * (np.pi / 2) ** (n - 1)
        V = V / (np.math.factorial(n - 1) * np.math.factorial(n - 2))
        return V

    # @classmethod
    def generate(self, r_in):
        """Generate flat phase-space configuration using 3n-4 random numbers"""
        n = self.npar
        self.K = np.zeros(n)
        self.M = np.zeros(n)
        # first (n-2) random numbers: r[0] ... r[n-1]
        B = self.generate_intermediates(r_in[:n - 2])
        Q = self.total_p
        for i in range(1, n):
            # want to continue from r[n-2] to r[3n-6]
            r = [r_in[2 * i + n - 4], r_in[2 * i + n - 3]]
            # r[n-1] to r[3n-5]
            print(self.M[i - 2], self.M[i - 1])
            p = self.decay_two_body_system(r, self.M[i - 1], self.M[i], self.particles[i - 1].m)
            print("p", p, "Q:", Q)
            Qnext = p[0]
            print("before:", self.particles[i - 1])
            self.particles[i - 1].set_p(p[1])
            print("mid:", self.particles[i - 1])
            self.particles[i - 1].set_p(p[1].boost_2(Q))
            print("after:", self.particles[i - 1])
            Qnext = p[0].boost_2(Q)
            Q = Qnext
        self.particles[n - 1].set_p(Q)

    @staticmethod
    def decay_two_body_system(r, m, m1, m2):
        """
        Decays two-body system using two-dimensional (random) parameters in r.
        Args:
            r:  (random) array in [0,1]^2: rescaled (cos theta, phi).
            m:  decay mass
            m1, m2:  masses of decayed systems (?)
        """
        cos_theta = 2 * r[0] - 1
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        phi = 2 * np.pi * r[1]
        if phi > np.pi:
            phi = phi - 2 * np.pi

        #    k_abs = np.sqrt (lambda (phs%m(i - 1)**2, phs%m(i)**2, phs%m_out(i - 1)**2)) &
        #         / (2. * phs%m(i - 1))

        p_space_abs = 4 * m * rho(m, m1, m2)
        p_space = np.array([np.cos(phi) * sin_theta,
                            np.sin(phi) * sin_theta,
                            cos_theta])
        p_space = p_space_abs * p_space
        p = []
        p.append(ParticleFourMom(np.concatenate([[np.sqrt(p_space_abs ** 2 + m1 ** 2)], p_space])))
        p.append(ParticleFourMom(np.concatenate([[np.sqrt(p_space_abs ** 2 + m2 ** 2)], -p_space])))
        return p

    def generate_intermediates(self, r):
        """Generates intermediate-mass systems for massive final states"""
        print(self)
        n = self.npar
        self.M[0] = self.total_p.mass()
        self.M[n - 1] = self.particles[n - 1].m
        self.K = self.calculate_k(r)
        for i in range(1, n - 1):
            self.M[i] = self.K[i]
            for j in range(i, self.npar):
                self.M[i] = self.M[i] + self.particles[j].m
        self.jac = self.K[0] ** (2 * n - 4) * 8 * rho(self.M[n - 2], self.particles[n - 1].m, self.particles[n - 2].m)
        for i in range(1, n - 1):
            self.jac = self.jac * rho(self.M[i - 1], self.M[i], self.particles[i - 1].m) / rho(self.K[i - 1], self.K[i], 0) * (
                        self.M[i] / self.K[i])

    def calculate_k(self, r):
        n = self.npar
        K = np.zeros(n)
        K[0] = self.M[0]
        for i in range(n):
            K[0] = K[0] - self.particles[i].m
        u = self.solve_for_u(r)
        for i in range(1, n - 1):
            K[i] = np.sqrt(u[i] * K[i - 1] ** 2)
        print("K", K)
        return K

    #   verified: does solve for u
    def solve_for_u(self, r):
        nbisect = 100
        n = self.npar
        u = np.zeros(n)
        for i in range(1, n - 1):
            xl = 0
            xr = 1
            if r[i - 1] == 1 or r[i - 1] == 0:
                u[i] = r[i - 1]
            else:
                for j in range(nbisect):
                    xmid = (xl + xr) / 2
                    f = self.f_rambo(xl, n - (i + 1)) - r[i - 1]
                    f_mid = self.f_rambo(xmid, n - (i + 1)) - r[i - 1]
                    if f * f_mid > 0:
                        xl = xmid
                    else:
                        xr = xmid
                    if np.isclose(xl - xr, 0):
                        break
                u[i] = xmid
        return u

    @staticmethod
    def f_rambo(u, n):
        # evaluates to (n+1-i)
        return (n + 1) * u ** n - n * u ** (n + 1)


def rho(M1, M2, m):
    if M1 != 0:
        rho = np.sqrt((1 - (M2 / M1 + m / M1) ** 2) * (1 - (M2 / M1 - m / M1) ** 2)) / 8
    else:
        raise ValueError("M1 shouldn't be 0?")
    # return sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M));
    return rho
