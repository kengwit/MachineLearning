from pyDOE import *
from scipy.stats.distributions import norm

design = lhs(4, samples=10)
means = [1, 2, 3, 4]
stdvs = [0.1, 0.5, 1, 0.25]
for i in range(0,4):
    design[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(design[:, i])