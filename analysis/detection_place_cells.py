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

# Preprocessing activities to be in range 0,1
## Will have to be changed to consider a cell and its opposite for each cell
def preprocessing(activities):
    processed_activites = []
    for act in iter(activities.T):
        if np.mean(act) < 0:
            new_act = standardize(-1*act)
        else :
            new_act = standardize(act)
        processed_activites.append(new_act)
    return np.array(processed_activites).T

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
    plt.imshow(explo_map.T, cmap='YlOrRd', origin='lower')
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
    plt.figure(figsize=(6, 3))
    #plt.imshow(map, cmap = 'YlOrRd', origin='lower', vmin=0, vmax=1)
    #plt.grid(visible=True, fillstyle='full') # Will have to work
    plt.pcolor(map, cmap= 'YlOrRd', edgecolors='gray', linewidths=1, linestyle='--', vmin=0, vmax=1)
    plt.gca().invert_yaxis()
    plt.show()

######################################################################################################################################

# Detecting place cells using the Peak method
def place_cells_detection_peak(path, resolution):
    res_states = preprocessing(load_reservoir_states(path))
    positions = load_positions(path)
    nb_neurons = len(res_states[0])
    explo_map = exploratory_map(resolution, positions)
    
    # Computing the actual peak for each neuron
    true_peaks = np.array([np.max(np.abs(activity_map(res_states.T[i], positions, explo_map, resolution))) for i in range(nb_neurons)])

    # Computing random peaks by shuffling the activity nb_shuffle times for a neuron
    def random_peaks(neuron_activity, nb_shuffles):
        peaks = np.zeros(nb_shuffles)
        for n in range(nb_shuffles):
            rd_activity = np.copy(neuron_activity)
            rd.shuffle(rd_activity) # in place
            peaks[n] = np.max(np.abs(activity_map(rd_activity, positions, explo_map, resolution)))
        return peaks
    
    # Computing the 99-percentile for each set of peaks
    nb_shuffle = 500

    peaks = np.array([np.concatenate((np.array([true_peaks[i]]), random_peaks(res_states.T[i], nb_shuffle))) for i in range(nb_neurons)])
    percentiles = [np.percentile(peaks[i], 99) for i in range(nb_neurons)]

    # Only keeping cells with a true peak over the 99-percentile of its peak set
    is_place_cell = [peaks[i,0] >= percentiles[i] for i in range(nb_neurons)]
    place_cells = np.argwhere(is_place_cell)
    return place_cells.ravel()

#########################################################################################################################################

# Detecting place cells using stability method TODO: to be fixed, seems to be irrelevant
def place_cells_detection_stability(resolution):
    #path = "/home/heloise/Mnémosyne/splitter-cells/trials/first_attempt/reservoir_states/"
    #path = "/home/heloise/Mnémosyne/splitter-cells/data/RR-LL/no_cues/reservoir_states/"
    path = "/home/heloise/Mnémosyne/splitter-cells/trials/mix/maze_other/"
    res_states = load_reservoir_states(path).T[:150].T
    #res_states = np.array([res_states.T[953], res_states.T[1457], res_states.T[323], res_states.T[267], res_states.T[1251]]).T
    ##res_states = np.array([res_states.T[81], res_states.T[1486], res_states.T[23], res_states.T[1201]]).T
    positions = load_positions(path)
    nb_neurons = len(res_states[0])

    # Defining halves of the trajectory
    positions_1, positions_2 = positions[:floor(len(positions)/2)], positions[floor(len(positions)/2):]
    res_states_1, res_states_2 = res_states[:floor(len(positions)/2)], res_states[floor(len(positions)/2):]
    # ... And their corresponding exploratory maps
    explo_map_1, explo_map_2 = exploratory_map(resolution, positions_1), exploratory_map(resolution, positions_2)
    
    # Actual activity maps for each neuron
    maps_1 = np.array([activity_map(state, positions_1, explo_map_1, resolution) for state in res_states_1.T])
    maps_2 = np.array([activity_map(state, positions_2, explo_map_2, resolution) for state in res_states_2.T])

    # Actual Pearson correlation coefficient between first section and second section maps
    true_coefs = np.array([pearsonr(maps_1[i].ravel(),maps_2[i].ravel()).statistic for i in range(nb_neurons)])

    # Computing coefficients for random maps by shuffling the activity nb_shuffle times for each neuron
    def random_coefs(m1, nb_shuffles):
        coefs = np.zeros(nb_shuffles)
        maps = rd.choices(maps_2, k=nb_shuffles)
        for n in range(nb_shuffles):
            coefs[n] = pearsonr(m1.ravel(), maps[n].ravel()).statistic
        return coefs
    
    # Computing the 99-percentile for each set of activities
    nb_shuffle = 300
    coefs = np.array([np.concatenate((np.array([true_coefs[i]]), random_coefs(maps_1[i], nb_shuffle))) for i in range(nb_neurons)])
    percentiles = [np.percentile(coefs[i], 95) for i in range(nb_neurons)]

    is_place_cell = [coefs[i,0] >= percentiles[i] for i in range(nb_neurons)]
    place_cells = np.argwhere(is_place_cell)
    return place_cells.ravel()

#################################################################################################################################

def place_cells_stability_filter(resolution):
    #path = "/home/heloise/Mnémosyne/splitter-cells/trials/first_attempt/reservoir_states/"
    #path = "/home/heloise/Mnémosyne/splitter-cells/data/RR-LL/no_cues/reservoir_states/"
    path = "/home/heloise/Mnémosyne/splitter-cells/trials/trained_maze_other__new_maze/"
    r = load_reservoir_states(path)
    res_states = r[450:1500]
    #res_states = np.array([res_states.T[953], res_states.T[1457], res_states.T[323], res_states.T[1251], res_states.T[267]]).T
    #res_states = np.array([res_states.T[81], res_states.T[1486], res_states.T[23], res_states.T[1201]]).T
    positions = load_positions(path)[450:1500]
    nb_neurons = len(res_states[0])

    def activity_variance(activity, resolution):
        activities = [[[] for j in range(resolution[1])] for i in range(resolution[0])]
        for i,pos in enumerate(positions):
            # Computing correspondance between position and indexes in the repartition matrix
            x = floor((pos[0] / 300) * resolution[0])
            y = floor((pos[1] / 500) * resolution[1])
            activities[x][y].append(activity[i])
        for i, l in enumerate(activities):
            for j, act in enumerate(l):
                if act == []:
                    activities[i][j] = nan # when it's a wall
                else:
                    activities[i][j] = np.var(act)
        activities = [[x for x in activities[i] if x==x] for i in range(len(activities))]
        return activities
    
    # Mean for inhomogeneous tabs
    def mean(tab):
        val = 0
        nb_elem = 0
        for x in iter(tab):
            for y in iter(x):
                nb_elem += 1
                val += y
        return val/nb_elem
    
    sum_variances = [-1*mean(activity_variance(state, resolution)) for state in res_states.T]        
    percentile = np.percentile(sum_variances, 99)
    is_place_cell = [sum_variances[i] >= percentile for i in range(nb_neurons)]
    place_cells = np.argwhere(is_place_cell)
    return place_cells.ravel()

#############################################################################################################################################

def draw_place_cell(resolution, radius):
    cell = np.zeros(resolution)
    middle = (floor(resolution[0]/2), floor(resolution[1]/2))
    for x in range(resolution[0]):
        for y in range(resolution[1]):
            if (abs(x-middle[0]) <= radius) and (abs(y-middle[1]) <= radius):
                cell[x,y] = 0.5
    cell[middle[0], middle[1]] = 1
    plt.imshow(cell.T, cmap='coolwarm', vmin=0, vmax=1, origin='lower')
    plt.show()
    return cell

# Radius defines the size of a place field
def place_cells_filter_specificity(path, resolution, radius):
    res_states = preprocessing(load_reservoir_states(path))
    positions = load_positions(path)
    nb_neurons = len(res_states[0])
    explo_map = exploratory_map(resolution, positions)

    def diff_mean(act_map, radius):
        pos = np.argmax(act_map)
        pos_max = [floor(pos/len(act_map[0])), pos%len(act_map[0])]
        map_field = np.zeros(resolution)
        #map_comp = np.zeros((resolution))
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                # In the place field
                if (pos_max[0]-radius <= i <= pos_max[0]+radius) and (pos_max[1]-radius <= j <= pos_max[1]+radius):
                    map_field[i,j] = act_map[i,j]
                # Outside
                #else:
                    #map_comp[i,j] = act_map[i,j]
        
        mf_non_zero = map_field[map_field > 0]
        mc_non_zero = act_map[act_map > 0]
        
        # Testing whether any array is empty (given that all values are now non zero)
        if not mf_non_zero.any():
            peak_field = 0
        else:
            peak_field = np.sum(mf_non_zero)
        if not mc_non_zero.any():
            peak_comp = 0
        else:
            peak_comp = np.sum(mc_non_zero)
        
        return [peak_field, peak_comp]
    

    # Array with each neurons mean activity across the maze
    activity_maps = np.array([activity_map(state, positions, explo_map, resolution) for state in res_states.T])
    delta_peaks = np.array([diff_mean(activity_maps[i], radius) for i in range(nb_neurons)])
    
    is_place_cell = [delta_peaks[i,0] > 0.7 * delta_peaks[i,1] for i in range(len(delta_peaks))]
    #is_place_cell = [delta_peaks[i,0] > 0.4 and delta_peaks[i,1] <= 0 for i in range(nb_neurons)]
    place_cells = np.argwhere(is_place_cell)
    return place_cells.ravel()

###################################################################################################

'''
Trying to detect border cells

'''

path = "data/RR-LL/no_cues/reservoir_states/"
pos = load_positions(path)
activity = load_reservoir_states(path).T[rd.randint(0,1000)]
resolution = (6,10)
explomap = exploratory_map(resolution, pos)
#plot_map(explomap)
#actmap = activity_map(activity, pos, explomap, resolution)
plot_activity_map(activity, pos, explomap, resolution)

#exploratory_map((6,10))
#print(walls_collision_percentage())
#show_place_fields((6,10))
#plot_activity_map(state, positions, exploratory_map((12,20), positions), (12,20))

#place_cells_peak = place_cells_detection_peak(path, (12,20))
#print(place_cells_peak)

#place_cells_stab = place_cells_detection_stability((12,20))
#print(place_cells_stab)

#place_cells_spec_fliltered = place_cells_filter_specificity(path, (12,20), 1)
#print(place_cells_spec_fliltered)

#pc_stab_fil = place_cells_stability_filter((36,60))
#print(pc_stab_fil)

#draw_place_cell((36,60), 2)