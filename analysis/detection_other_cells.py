import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from math import *

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

# Returns an array with the amount of times the bot entered each bin
def exploratory_map(resolution, positions):
    movement_repartition = np.zeros(resolution)

    for pos in iter(positions):
        # Computing correspondance between position and indexes in the repartition matrix
        x = floor((pos[0] / 300) * resolution[0])
        y = floor((pos[1] / 500) * resolution[1])
        movement_repartition[x,y] += 1
    return movement_repartition

def plot_map(explo_map):
    # Normalizing the array
    explo_map /= np.std(explo_map)
    plt.imshow(explo_map.T, cmap='inferno', origin='lower')
    plt.show()

# Average activity in each bin for one neuron
def activity_map(neuron_activity, positions, explo_map, resolution):
    map = np.zeros(resolution)
    for i,pos in enumerate(positions):
        # Computing correspondance between position and indexes in the exploration matrix
        x = floor((pos[0] / 300) * resolution[0])
        y = floor((pos[1] / 500) * resolution[1])
        if explo_map[x,y] != 0:
            map[x,y] += neuron_activity[i] / explo_map[x,y]
        else:
            map[x,y] = 0
    map
    return map

# from matplotlib.colors import Normalize
def plot_activity_map(neuron, positions, explo_map, resolution):
    map = activity_map(neuron, positions, explo_map, resolution)
    plt.imshow(map.T, cmap = 'bwr', origin='lower', vmin=-1, vmax=1)
    plt.show()

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

def euclidian_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Given a list of corners positions, determins the corner score of a field
def corner_score_field(peak_field, corners):
    distances = np.array([euclidian_distance(peak_field, corner) for corner in iter(corners)])
    distance_centroid = euclidian_distance(peak_field, (150,250))
    dist_min = distances[np.argmin(distances)] # sur une liste, so is ok
    return (distance_centroid - dist_min) / (distance_centroid + dist_min)

from scipy.ndimage import label, generate_binary_structure
# Given a list of corners positions, determins the corner score of a cell
# /!\ the centroïd is fixed at 150,250 /!\
def corner_score_cell(resolution, actmap, corners):
    peak_map = np.max(actmap)
    filtered_map = np.array([act > peak_map * 0.3 for act in np.nditer(actmap)])
    s = generate_binary_structure(2,2)
    labelled_map, n_fields = label(actmap, structure=s)

    # Filtered fields maxima positions
    peaks_positions = []
    for i in range(1, n_fields+1):
        field = actmap[filtered_map == i]
        # Fields peak positions
        peak = np.argmax(field)
        pos_max = [floor(peak/len(actmap[0])), peak%len(actmap[0])]
        # Correspondance in the maze
        dim = [300,500]
        true_pos_max = [coor*resolution[i] + dim[i]/(2*resolution[i]) for i,coor in enumerate(pos_max)]
        peaks_positions.append(true_pos_max)
    
    # TODO: have a way to detect les corners


##############################################################################################################

def head_direction_detection():
    pass