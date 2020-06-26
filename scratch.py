import numpy as np
import pickle

from afpo import AFPO

# np.random.seed(0)
afpo = AFPO(gens=10, popsize=10)
best, fit = afpo.run()
best.CPPN_summary()