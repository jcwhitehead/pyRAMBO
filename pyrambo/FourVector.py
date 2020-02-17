import numpy as np

class FourVec(object):
    """Represents an arbitrary four-vector.

    Binary operations include:
       - addition, subtraction, inversion: overload their respective operators.
       - dot: four-vector dot-product with respect to the Minkowski metric (+--- convention).
       - restframe: returns the four-vector in its own rest-frame.
       - restframe_boost: boosts the four-vector into the rest-frame of another four-vector.
       - restframe_boost_lambda: returns the Lorentz matrix representing a boost into the rest-frame of the four-vector.
       - restframe_boost_lambda_inv: returns the inverse of the above.
       """

    def __init__(self, p):
        """
        Initialises FourVec class.

        Args:
            p: four-momentum as a list or array.
        """
        self.p4 = np.array(p)
        self.p3vec = np.array(p[1:4])
        self.lightlike = False
        self.timelike = False
        self.spacelike = False
        self.normsq = self.dot(self)
        if np.isclose(self.normsq, 0):
            self.lightlike = True
        elif self.normsq > 0:
            self.timelike = True
        elif self.normsq < 0:
            self.spacelike = True
        self.norm = np.sqrt(abs(self.normsq))

    def dot(self, other):
        """Calculates the four-vector (Minkowski) dot-product; signature (+---).

        Args:
            other: another four vector.

        Returns:
            Inner product of 'self' with 'other'.
        """
        return self.p4[0] * other.p4[0] - np.dot(self.p4[1:], other.p4[1:])

    def restframe(self, nvec=np.array([1, 0, 0])):
        """

        Args:
            nvec:

        Returns:

        """
        nvec = FourVec.normalise(nvec)
        m = self.norm
        if self.timelike:
            return np.array([m, 0, 0, 0])
        elif self.spacelike:
            # arbitrary choice to align with x-axis
            return np.concatenate([[0], m * np.array(nvec)])
        elif self.lightlike:
            raise ValueError("A lightlike four-vector has no rest-frame")

    def restframe_boost(self, other, sgn=1):
        """
        Boosts self into the rest-frame of other.

        Args:
            other:
            sgn:

        Returns:
            FourVec of result.
        """
        abs_ref = np.sqrt(abs(other.dot(other)))
        bvec = -np.sign(sgn) * other.p3vec / abs_ref
        gamma = other.p4[0] / abs_ref
        a = 1 / (1 + gamma)
        bp = np.dot(self.p3vec, bvec)
        outE = gamma * self.p4[0] + bp
        out3v = self.p3vec + bvec * (self.p4[0] + a * bp)
        return FourVec(np.concatenate([[outE], out3v]))

    # slightly different implementation, but amounts to the inverse of the above.
    def boost_2(self, other, sgn=1):
        """

        Args:
            other:
            sgn:

        Returns:

        """
        if np.allclose(other.p3vec, 0):
            return FourVec(self.p4)
        b = np.sign(sgn) * other.p3vec / other.p4[0]

        b_sq = np.dot(b, b)
        g = 1 / np.sqrt(1 - b_sq)
        g_sq = (g - 1) / b_sq
        bv = np.dot(b, self.p3vec)
        outE = g * self.p4[0] + g * bv
        outp3 = self.p3vec + g_sq * bv * b + g * b * self.p4[0]
        return FourVec(np.concatenate([[outE], outp3]))

    def restframe_boost_inv(self, other):
        """

        Args:
            other:

        Returns:

        """
        return self.restframe_boost(other, sgn=-1)

    def __add__(self, other):
        """
        Adds four-vectors.

        Args:
            other:

        Returns:
            self + other (FourVec)
        """
        psum = self.p4 + other.p4
        return FourVec(psum)

    def __radd__(self, other):
        """
        Right-addition of four-vector with other; necessary for sum() to work.

        Args:
            other:

        Returns:

        """
        # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        Four-vector subtraction.

        Args:
            other:

        Returns:

        """
        pdiff = self.p4 - other.p4
        return FourVec(pdiff)

    def __neg__(self):
        """
        Four-vector inversion (p -> -p).

        Returns:
            -p (FourVec)
        """
        return FourVec(-self.p4)

    def __repr__(self):
        return "FourVec(np.array({0},{1},{2},{3}))".format(self.p4[0], self.p4[1], self.p4[2], self.p4[3])

    def __str__(self):
        return "({0},{1},{2},{3})".format(self.p4[0], self.p4[1], self.p4[2], self.p4[3])

    def mass(self):
        """
        Calculates Minkowski norm of four-vector (= mass for physical particles through p^2 = m^2).

        Returns:
            mass
        """
        return self.norm

    def restframe_boost_lambda(self, sgn=1):
        """
        Returns Lorentz boost matrix that boosts into the rest-frame of the 4-vector.

        Args:
            sgn (int): +1 for forwards transformation; -1 for inverse.
        """
        rma2 = self.dot(self)
        rma = np.sqrt(rma2)
        gam = self.p4[0] / rma
        v = np.sign(sgn) * self.p4 / self.p4[0]
        v2 = np.dot(v[1:], v[1:])
        bmat = np.zeros((4, 4))
        bmat[0, 0] = gam
        bmat[0, 1:] = -gam * v[1:]
        bmat[1:, 0] = -gam * v[1:]
        bmat[1:, 1:] = (gam - 1) * np.outer(v[1:], v[1:]) / v2 + np.identity(3)
        return bmat

    def restframe_boost_lambda_inv(self):
        return self.restframe_boost_lambda(sgn=-1)

    @staticmethod
    def normalise(vec):
        v = np.array(vec)
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0):
            return np.zeros(np.shape(vec))
        return v / norm