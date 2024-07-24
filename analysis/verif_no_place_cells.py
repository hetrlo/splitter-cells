import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from math import *
import sys
sys.path.append('../splitter-cells')

# Functions to load data
def load_positions(path):
    return np.load(path + 'positions.npy')


def load_reservoir_states(path):
    return np.load(path + 'reservoir_states.npy')


def load_orientations(path):
    return np.load(path + 'output.npy')

def load_sensors(path):
    return np.load(path + 'input.npy')

# Standardization of an array : values will be in range [0,1]
def standardize(array):
    array = np.abs(array)
    max_arr = np.max(array)
    min_arr = np.min(array)
    if max_arr==0:
        return array
    if min_arr == max_arr:
        return array / max_arr
    return (array - min_arr) / (max_arr - min_arr)

# Preprocessing activities to be in range 0,1
def preprocessing(activities):
    processed_activites = []
    for act in iter(activities.T):
        if np.mean(act) < 0:
            new_act = standardize(-1*act)
        else :
            new_act = standardize(act)
        processed_activites.append(new_act)
    return np.array(processed_activites).T

path = "/home/heloise/Mnémosyne/splitter-cells-results/traj/verif place cells/50,100/"
activities = load_reservoir_states(path)

neurons = rd.sample(range(0,999), 5)
neurons = [0,1,2,3,4]
for neuron in neurons:
    plt.plot(activities.T[neuron])
plt.show()

# Trying to conclude with our activities
# Computes the percentage of time a neuron was active
def percentage_activation(activity, threshold_act):
    perc_activ = np.count_nonzero(activity[activity > threshold_act])
    return perc_activ / len(activity)

def place_cell_like(activities, threshold_act, threshold_time):
    # Processing data
    activities = preprocessing(load_reservoir_states(path))
    nb_neurons = len(activities[0])
    scores = np.array([percentage_activation(act, threshold_act) for act in iter(activities.T)])
    is_cell = [scores[i] >= threshold_time for i in range(nb_neurons)]
    place_cells = np.argwhere(is_cell)
    return place_cells.ravel()