"""functions to convert activations into EKG and compute fitness"""

import numpy as np
import pickle

def convert_action_to_charge(activations):

    # # testing
    # a = [5, 2, 3, 10, 15, 11, 3, 2, 1]
    # activations = np.reshape(a, (len(a), 1))

    pos_charge_dict = {} # empty dictionary - at each time point, tells which cells are excited

    for i in range(1,np.max(activations)+1):
        pos_charge_dict[i-1] = []
        for j in range(len(activations)):
            if activations[j,0] == i:
                pos_charge_dict[i-1].append(j)

    return pos_charge_dict

def calc_EKG(dipole, lead_distances):

    pass

f = open('activation_matrix.p', 'rb')
activations = pickle.load(f)
f.close()

f = open('lead_coords.p', 'rb')
lead_distances = pickle.load(f)
f.close()

# convert_action_to_charge(activations)
calc_EKG([3], lead_distances)