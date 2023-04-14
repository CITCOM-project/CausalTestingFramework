from scipy.stats import rv_discrete


class EnumGen(rv_discrete):
    def __init__(self, dt: Enum):
        self.dt = dict(enumerate(dt, 1))
        self.inverse_dt = {v: k for k, v in self.dt.items()}

    def ppf(self, q, *args, **kwds):
        return np.vectorize(self.dt.get)(np.ceil(len(self.dt) * q))

    def cdf(self, q, *args, **kwds):
        return np.vectorize(self.inverse_dt.get)(q) / len(Car)
