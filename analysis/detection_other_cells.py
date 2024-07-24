import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from math import *
from maze import *
from single_cell_analysis import plot_hippocampal_cells

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
    plt.imshow(explo_map.T, cmap='inferno', origin='lower')
    plt.show()

def plot_actmap(actmap):
    plt.imshow(actmap.T, cmap = 'bwr', origin='lower', vmin=-1, vmax=1)
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


####################################################################################

# Set of corners for each maze (TODO: do that automatically)

# maze
maze = Maze()
convex_walls = maze.walls[:4].ravel()
concave_walls = maze.walls[4:12].ravel()

maze_convex_corners = list(set([(convex_walls[2*i], convex_walls[2*i+1]) for i in range(int(len(convex_walls)//2))]))
maze_concave_corners = list(set([(concave_walls[2*i], concave_walls[2*i+1]) for i in range(int(len(concave_walls)//2))]))

# random_walls => fait à la main, faudra faire ça mieux
random_walls_convex_corners = [(200,450), (100,275), (100,325), (200,300), (250,300), (250,350),
                               (200,350), (50,160), (150,50), (250,50), (250,150), (200,150), (150,100)]
random_walls_concave_corners = [(0,500), (200,500), (500,450), (0,275), (0,325), (300,0), (0,160), 
                                (50,0), (200,100)]

# empty
empty_concave_corners = maze_concave_corners # t'façon y'a que ça.

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
# peak_field is the position of the peak in said field
def corner_score_field(peak_field, corners):
    distances = np.array([euclidian_distance(peak_field, corner) for corner in iter(corners)])
    distance_centroid = euclidian_distance(peak_field, (150,250)) # ne marche que pour empty ptdr
    dist_min = distances[np.argmin(distances)] # sur une liste, so is ok
    return (distance_centroid - dist_min) / (distance_centroid + dist_min)

from scipy.ndimage import label, generate_binary_structure
# Given a list of corners positions, determins the corner score of a cell
# /!\ the centroïd is fixed at 150,250 /!\
# Based on: The subiculum encodes environmental geometry
def corner_score_cell(resolution, actmap, corners):
    filter = np.array([[actmap[i,j] > 0.7 for j in range(len(actmap[0]))] for i in range(len(actmap))])
    filtered_map = actmap * filter
    s = [[0,0,1,0,0],
         [0,1,1,1,0],
         [1,1,1,1,1],
         [0,1,1,1,0],
         [0,0,1,0,0]]
    labelled_map, n_fields = label(filtered_map)
    # Filtered fields maxima positions
    peaks_positions = []
    for n in range(1, n_fields+1):
        field = np.zeros(resolution)
        size_field = 0
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                if labelled_map[i,j] == n:
                    field[i,j] = actmap[i,j]
                    size_field += 1

        # A place field can't take bigger than 10% of the maze
        if size_field < resolution[0]*resolution[1]*0.1:
            plt.imshow(field.T, origin='lower', cmap='YlOrRd')
            plt.show()
            # Fields peak positions
            peak = np.argmax(field)
            pos_max = [floor(peak/len(actmap[0])), peak%len(actmap[0])]
            # Correspondance in the maze
            dim = [300,500]
            true_pos_max = [int(coor*resolution[i] + dim[i]/(2*resolution[i])) for i,coor in enumerate(pos_max)]
            # pos_max is the index in the act map, true_pos_max is the position in the maze
            peaks_positions.append((pos_max, true_pos_max))
    
    # Two case depending on wether the cell has less fields than the number of corners or not
    def corner_score(peaks_positions, corners):
        # Defining constants
        k = len(corners)
        n = len(peaks_positions)

        # Ordering positions by activity
        def act_order(pos):
            i,j = pos[0]
            return actmap[i,j]
        
        peaks_positions.sort(key=act_order)
        peaks_positions = np.array(peaks_positions)
        corner_scores = np.array([corner_score_field(pos[1], corners) for pos in iter(peaks_positions)])
        if n <= k:
            score = np.sum(corner_scores) / k
        else:
            # The extra fields act as a penality to avoid cells with too many fields
            score = (np.sum(corner_scores[:k]) - np.sum(np.abs(corner_scores[k:]))) / k
        return score
    return corner_score(peaks_positions, corners)

def corner_cells_classification(path, resolution, corners):
    # Processing data
    activities = preprocessing(load_reservoir_states(path))
    positions = load_positions(path)
    nb_neurons = len(activities[0])
    exp_map = exploratory_map(resolution,positions)
    act_maps = np.array([activity_map(act, positions, exp_map, resolution) for act in iter(activities.T)])

    # Computing the actual score for each cell
    true_scores = np.array([corner_score_cell(resolution, map, corners) for map in iter(act_maps)])

    # Generating random scores
    def random_scores(activity, nb_shuffles):
        random_peaks = []
        for i in range(nb_shuffles):
            rd_activity = np.copy(activity)
            rd.shuffle(rd_activity) # in place
            rd_map = activity_map(rd_activity, positions, exp_map, resolution)
            random_peaks.append(corner_score_cell(resolution, rd_map, corners))
        return np.array(random_peaks)
    
        # Computing the 99-percentile for each set of peaks
    nb_shuffle = 50

    scores = []
    percentiles = []
    for i in range(nb_neurons):
        scores.append(np.concatenate((np.array([true_scores[i]]), random_scores(activities.T[i], nb_shuffle))))
        percentiles.append(np.percentile(scores[i], 99))
        if i%100 == 0:
            print("Analysis %d%% done" %int(i/nb_neurons*100))
    scores = np.array(scores)

    # Only keeping cells with a true peak over the 99-percentile of its peak set
    is_corner_cell = [scores[i,0] >= percentiles[i] for i in range(nb_neurons)]
    corner_cells = np.argwhere(is_corner_cell)
    return corner_cells.ravel()

    
##############################################################################################################

def head_direction_detection():
    pass

###############################################################################################################

# Testing corner cells classification on empty maze:
path = "/home/heloise/Mnémosyne/splitter-cells-results/traj/esn/RR-LL/"
resolution = (6,10)
activities = preprocessing(load_reservoir_states(path)).T[:20].T
positions = load_positions(path)
explo_map = exploratory_map(resolution, positions)
#plot_map(explo_map)

corner_cells = corner_cells_classification(path, (6,10), maze_concave_corners)
print("Corner cells:", corner_cells)

#plot_activity_map(activities.T[81], positions, explo_map, resolution)
#plot_hippocampal_cells(path, [7, 55, 77, 81])