import numpy as np
from pyrambo.FourVector import FourVec
from pyrambo.ParticleFourMom import ParticleFourMom

class PhaseSpace(object):
    """Abstract phase-space class.

    Attributes:
        npar (int):
        total_p (FourVec):
        masses (ndarray):
        particles (list):
        flux: [not yet implemented]
        vol: [not yet implemented]
        jac: [not yet implemented]
        w: [not fully implemented]
    """

    def __init__(self, npar, totalmom, massarray=None):
        """
        Initialises PhaseSpace class.

        Args:
            npar (int): total number of particles.
            totalmom:
            massarray:
        """
        np.seterr(invalid='raise')
        self.npar = npar
        self.total_p = FourVec(totalmom)
        if len(massarray) != npar:
            raise ValueError("Dimension of massarray must match npar.")
        if not massarray:
            massarray = [0] * npar
        self.masses = np.array(massarray)
        particles = []
        for i, m in enumerate(self.masses):
            particles.append(ParticleFourMom.atrest(m))
        self.particles = particles
        self.flux = None
        self.vol = None
        self.jac = None

    def __str__(self):
        A = "Npar:  {npar}; \n Q^2:  {mom}; \n; Vol: {vol}; \n; Jac: {jac} \n"
        return A.format(npar=self.npar, mom=self.total_p, vol=self.vol, jac=self.jac)

    def p_i(self, i):
        return self.particles[i]

    def weight(self):
        self.w = self.jac * self.vol  # *self.flux?
        return self.w
