
from utils import MovingAverage

class SE_Generator(MovingAverage):
    pass

    # def se2al(self, se):
    #     # given the current spectral efficiency (SE), return the aggregation level (AL)
    #      = (se - self.lims[0])
    #     delta = (self.lims[1] - self.lims[0])
    #     if  >= delta * .8:
    #         return 1
    #     if  >= delta * .6:
    #         return 2
    #     if  >= delta * .3:
    #         return 4
    #     return 8
    #
