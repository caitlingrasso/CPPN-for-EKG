import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import time

from util import write_to_txt, plot_ekg, load_data, load_txt_file
from evaluate_EKG import evaluate_EKG, calc_EKG, convert_action_to_charge, convert_activation_to_charge
from afpo import AFPO
from cppn import CPPN
from genome import Genome


# Loading from text file example
load_txt_file('data/leadDistances.txt')

# to verify that data was loaded correctly...
leadDistances = load_data('data/leadDistances.p')
print(leadDistances)

# Generating random cppns
cppn = CPPN(novel=True)
cppn.print()

# Test afpo run
start_time = time.time()

afpo = AFPO(gens=2, popsize=2)
best, fits_per_gen = afpo.run()

print('--', time.time()-start_time, 'seconds --')

print(best.fitness)
best.CPPN_summary()
best.plot_EKG(save=True)
