from pyrambo.FourVector import FourVec

class ParticleFourMom(FourVec):
    """Simple extension to FourVec class to include mass."""

    def __init__(self, p, m=None):
        # change if migrating to Python 3 to super().__init__(p)
        super().__init__(p)
        self.E = p[0]
        if m:
            self.m = m
        else:
            self.m = self.norm

    def set_p(self, pin):
        m = self.m
        if isinstance(pin, FourVec):
            pin = pin.p4
        super(ParticleFourMom, self).__init__(pin)
        self.m = m
        self.E = pin[0]

    @classmethod
    def atrest(cls, m):
        """Constructs a particle with mass m, at rest"""
        return cls([m, 0, 0, 0])

    #    @classmethod
    #    def notonshell(self, m, p):
    #        """Constructs a particle with mass m and momentum p, not necessarily on-shell"""
    #        super(ParticleFourMom, self).__init__(p)
    #        self.E = p[0]
    #        self.m = m

    def __repr__(self):
        return "ParticleFourMom(np.array({0},{1},{2},{3}))".format(self.p4[0], self.p4[1], self.p4[2], self.p4[3])