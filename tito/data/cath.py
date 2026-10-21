from tito.data.octopeptide import Octopeptide, LaggedOctopeptide

class CATH(Octopeptide):
    pass

class LaggedCATH(LaggedOctopeptide):
    def __init__(self, *args, max_lag=10_000.0, **kwargs):
        super().__init__(*args, max_lag=max_lag, **kwargs)
