"""functions to convert activations into EKG and compute fitness"""

import numpy as np
import pickle

def convert_action_to_charge(activations):

    # testing
    # a = [5, 2, 3, 10, 15, 11, 3, 2, 1]
    # activations = np.reshape(a, (len(a), 1))

    pos_charge_dict = {} # empty dictionary - at each time point, tells which cells are excited

    for i in range(1,np.max(activations)+1):
        pos_charge_dict[i-1] = []
        for j in range(len(activations)):
            if activations[j,0] == i:
                pos_charge_dict[i-1].append(j)

    diPole = get_diPole_matrix(pos_charge_dict, len(activations))

    return diPole


def get_diPole_matrix(pos_charge_dict, n_cells):
    # diPole matrix = the contribution of each cell's charge at each time step

    diPole = np.zeros((n_cells, len(pos_charge_dict)), dtype=float)

    for t in range(len(pos_charge_dict)):
        excited_cells = pos_charge_dict[t]
        for cell in excited_cells:
            diPole[cell, t] = 0.5
            if t!=0:
                diPole[cell, t-1] = -0.5

    return diPole


def calc_EKG(diPole, lead_distances):

    calculated12Lead = np.zeros((diPole.shape[1], 12), dtype=float)

    # distances from each cell to the 10 different leads
    RAdistances = lead_distances[:, 0]
    LAdistances = lead_distances[:, 1]
    LLdistances = lead_distances[:, 3]
    v1distances = lead_distances[:, 4]
    v2distances = lead_distances[:, 5]
    v3distances = lead_distances[:, 6]
    v4distances = lead_distances[:, 7]
    v5distances = lead_distances[:, 8]
    v6distances = lead_distances[:, 9]

    for i in range(4, diPole.shape[1]):  # number of time steps

        # Eithoven
        calculated12Lead[i, 0] = np.sum(diPole[:,i]/(LAdistances**2)) - np.sum(diPole[:,i]/(RAdistances**2)) #I
        calculated12Lead[i, 1] = -(np.sum(diPole[:,i]/(LLdistances**2))-np.sum(diPole[:,i]/(RAdistances**2))) #II
        calculated12Lead[i, 2] = -(np.sum(diPole[:,i]/LLdistances**2)-np.sum(diPole[:,i]/LAdistances**2)) #III
        # Goldberg
        calculated12Lead[i, 3] = np.sum(diPole[:, i]/RAdistances**2)-.5 * ((np.sum(diPole[:, i]/LAdistances**2))+np.sum(diPole[:, i]/LLdistances**2)) #AVR
        calculated12Lead[i, 4] = np.sum(diPole[:, i]/LAdistances**2)-.5 * (np.sum(diPole[:, i]/LLdistances**2)+np.sum(diPole[:, i]/RAdistances**2)) #AVL
        calculated12Lead[i, 5] = -(np.sum(diPole[:, i]/LLdistances**2)-.5 * (np.sum(diPole[:, i]/LAdistances**2)+np.sum(diPole[:, i]/RAdistances**2))) #AVF
        # Wilson
        calculated12Lead[i, 6] = np.sum(diPole[:, i]/v1distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V1
        calculated12Lead[i, 7] = np.sum(diPole[:, i]/v2distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V2
        calculated12Lead[i, 8] = np.sum(diPole[:, i]/v3distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V3
        calculated12Lead[i, 9] = np.sum(diPole[:, i]/v4distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V4
        calculated12Lead[i, 10] = np.sum(diPole[:, i]/v5distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V5
        calculated12Lead[i, 11] = np.sum(diPole[:, i]/v6distances**2)-0.333 * (np.sum(diPole[:, i]/LAdistances**2)+(np.sum(diPole[:, i]/RAdistances**2))+(np.sum(diPole[:, i]/LLdistances**2))) # V6

    return calculated12Lead

#TODO: normalize time component
#TODO: root mean squared correlation between calculated and actual EKG

f = open('activation_matrix.p', 'rb')
activations = pickle.load(f)
f.close()

f = open('lead_distances.p', 'rb')
lead_distances = pickle.load(f)
f.close()

f = open('calculated12Lead.p', 'rb')
calculated12Lead_actual = pickle.load(f)
f.close()

diPole = convert_action_to_charge(activations)
calculated12Lead = calc_EKG(diPole, lead_distances)

print(calculated12Lead[0:9, :])
print('------------------------------------------------------------------------')
print(calculated12Lead_actual[0:9, :])

# print(np.allclose(calculated12Lead, calculated12Lead_actual))