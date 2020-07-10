"""functions to convert activations into EKG and compute fitness"""

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

from util import load_data, normalize, smooth, pickle_file

def evaluate_EKG(activations, lead_distances, answerEKG):
    """Calculates root mean squared error between actual EKG and EKG produced from input activation matrix"""

    calculated12Lead = calc_EKG(activations, lead_distances)

    calculated12Lead = normalize(calculated12Lead)

    calculated12Lead, answerEKG = match_time_scales(calculated12Lead, answerEKG)

    rmse = mean_squared_error(answerEKG, smooth_data(calculated12Lead), squared=False)
    # rmse = mean_squared_error(answerEKG, calculated12Lead, squared=False)  # without smoothing data

    return rmse


def calc_EKG(activations, lead_distances):
    """Calculates EKG signals from input activation matrix"""

    diPole = convert_activation_to_charge(activations)

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

def convert_activation_to_charge(activations):
    """Converts activation matrix to diPole matrix - each cell's charge at each time step"""
    activations = activations.astype(int) - 1
    cells = np.indices(activations.shape)
    diPole = np.zeros((len(activations), np.max(activations.astype(int))+1), dtype=float)

    diPole[cells[0][activations>=0], activations[activations>=0]] = 0.5
    diPole[cells[0][activations>0], activations[activations>0]-1] = -0.5

    return diPole

def convert_action_to_charge(activations):
    """Converts activation matrix to positive charge matrix"""
    pos_charge_dict = {} # empty dictionary - at each time point, tells which cells are excited
    # key = time step
    # value = list of indices of cells in the activation matrix that are excited at that time step

    for i in range(1,int(np.max(activations))+1):
        pos_charge_dict[i-1] = []
        for j in range(len(activations)):
            if activations[j,0] == i:
                pos_charge_dict[i-1].append(j)

    diPole = get_diPole_matrix(pos_charge_dict, len(activations))

    return diPole

def get_diPole_matrix(pos_charge_dict, n_cells):
    """produces diPole matrix - the contribution of each cell's charge at each time step"""

    diPole = np.zeros((n_cells, len(pos_charge_dict)), dtype=float)

    for t in range(len(pos_charge_dict)):
        excited_cells = pos_charge_dict[t]
        for cell in excited_cells:
            diPole[cell, t] = 0.5
            if t!=0:
                diPole[cell, t-1] = -0.5

    return diPole

def match_time_scales(series1, series2):
    """interpolates shorter time series to match number of samples"""
    if len(series1) < len(series2):
        factor = round(len(series2) / len(series1))
        series1_temp = np.zeros(series2.shape)
        for i in range(series1.shape[1]):  # number of leads
            y = series1[:, i]
            x = np.arange(len(series1))
            x = x * factor
            xvals = np.arange(len(series2))
            series1_temp[:, i] = np.interp(xvals, x, y)
        return series1_temp, series2
    elif len(series1) > len(series2):
        factor = round(len(series1) / len(series2))
        series2_temp = np.zeros(series1.shape)
        for i in range(series2.shape[1]):  # number of leads
            y = series2[:, i]
            x = np.arange(len(series2))
            x = x * factor
            xvals = np.arange(len(series1))
            series2_temp[:, i] = np.interp(xvals, x, y)
        return series1, series2_temp
    else:
        return series1, series2

def smooth_data(series):
    """smooths times series data using https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    series_smoothed = np.zeros(series.shape)
    for i in range(series.shape[1]):  # number of leads
        y = smooth(series[:, i])
        series_smoothed[:, i] = y[0:len(series)]
    return series_smoothed