"""
An analysis of the bot's trajectory, to quantify its exploration of the maze

"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

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
    

# Computes the percentage of time the bot touches the walls
def walls_collision_percentage():
    sensitivity = 0.05 # sensitivity of the sensors

    path = "/home/heloise/Mnémosyne/splitter-cells/trials/states/"
    sensors = load_sensors(path)
    len_record = len(sensors)
    print(len_record)

    # Determins whether the bot was in a wall at a given step
    def is_collision(sensors_values):
        collision = False
        for val in iter(sensors_values):
            collision = collision or (val < sensitivity)
        return collision
    
    collision = np.array([is_collision(sensors[i]) for i in range(len_record)])
    nb_collisions = np.count_nonzero(collision)
    return (nb_collisions / len_record) * 100

# Trying to fit an esn to compute position from the activity
def preprocess_positions(resolution, positions):
    discrete_positions = np.array([[0 for j in range(resolution[0]+resolution[1])] for i in range(len(positions))])
    for i,pos in enumerate(positions):
        # Computing correspondance between position and indexes in the repartition matrix
        x = floor((pos[0] / 300) * resolution[0])
        y = floor((pos[1] / 500) * resolution[1])
        discrete_positions[i,x] = 1
        discrete_positions[i,y+resolution[0]] = 1
    return discrete_positions

def process_positions(resolution, positions):
    discrete_x = np.array([[0 for j in range(resolution[0])] for i in range(len(positions))])
    discrete_y = np.array([[0 for j in range(resolution[1])] for i in range(len(positions))])
    for i,pos in enumerate(positions):
        # Computing correspondance between position and indexes in the repartition matrix
        x = floor((pos[0] / 300) * resolution[0])
        y = floor((pos[1] / 500) * resolution[1])
        discrete_x[i,x] = 1
        discrete_y[i,y] = 1
    return discrete_x, discrete_y

def plot_comparison_pred_test(resolution, pos_pred, pos_test):
    x_pred, y_pred, x_test, y_test = [], [], [], []
    nonsense_count = 0
    for (pred, pos) in iter(zip(pos_pred, pos_test)):
        pr, po = np.argwhere(pred).ravel(), np.argwhere(pos).ravel()
        if (len(pr) < 2):
            #print("Panic panic, mauvaise prédiction !")
            nonsense_count += 1
            continue
        # Prediction non sense now removed
        else:
            x_pred.append(pr[0])
            y_pred.append(pr[-1] - resolution[0])
            x_test.append(po[0])
            y_test.append(po[-1] - resolution[0])
    # Print percentage of nonsense values
    print("Nonsensical predictions percentage:", 
        nonsense_count / len(pos_pred) * 100)
    # Plotting comparisons for x and y
    x_pred, x_test = np.array(x_pred), np.array(x_test)
    plt.plot(x_pred, color='blue')
    plt.plot(x_test, color='orange', linestyle=":")
    #plt.plot(np.abs(x_pred-x_test), color='black')
    plt.show()
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    plt.plot(y_pred, color='blue')
    plt.plot(y_test, color='orange', linestyle=":")
    #plt.plot(np.abs(y_pred-y_test), color='black')
    plt.show()

def plot_pred_pos(xpred, ypred, xtest, ytest):
    x_pred, y_pred, x_test, y_test = [], [], [], []
    nonsense_count = 0
    for xp,yp,xt,yt in iter(zip(xpred, ypred, xtest, ytest)):
        xpi, ypi, xti, yti = np.argwhere(xp).ravel(), np.argwhere(yp).ravel(),np.argwhere(xt).ravel(), np.argwhere(yt).ravel()
        if (len(xpi) < 1) or (len(ypi)) < 1:
            nonsense_count += 1
            continue
        else:
            x_pred.append(xpi[0])
            y_pred.append(ypi[0])
            x_test.append(xti[0])
            y_test.append(yti[0])
    # Print percentage of nonsense values
    print("Nonsensical predictions percentage:", 
        nonsense_count / len(ypred) * 100)
    # Plotting comparisons for x and y
    x_pred, x_test = np.array(x_pred), np.array(x_test)
    plt.plot(x_pred, color='blue')
    plt.plot(x_test, color='orange', linestyle=":")
    #plt.plot(np.abs(x_pred-x_test), color='black')
    plt.show()
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    plt.plot(y_pred, color='blue')
    plt.plot(y_test, color='orange', linestyle=":")
    #plt.plot(np.abs(y_pred-y_test), color='black')
    plt.show()

def position_from_activity(resolution, path, nb_train):
    activities = load_reservoir_states(path)
    positions = load_positions(path)

    # Preprocessing positions : (0s and two 1s to indicate x and y)
    (disc_pos_x, disc_pos_y) = process_positions(resolution, positions)
    # Train and test splitting
    act_train, pos_train_x, pos_train_y = activities[:nb_train], disc_pos_x[:nb_train], disc_pos_y[:nb_train]
    act_test, pos_test_x, pos_test_y = activities[nb_train:], disc_pos_x[nb_train:], disc_pos_y[nb_train:]
    # Standardizing activity
    scaler = preprocessing.StandardScaler().fit(act_train)
    act_train = scaler.transform(act_train)
    act_test = scaler.transform(act_test)

    classifier_x = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(50,),
                                random_state=1, max_iter=200)
    classifier_x.fit(act_train, pos_train_x)

    classifier_y = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(20,20),
                                random_state=1, max_iter=200)
    classifier_y.fit(act_train, pos_train_y)

    # Classification
    pred_x = classifier_x.predict(act_test)
    pred_y = classifier_y.predict(act_test)
    print("Classifier x score:", classifier_x.score(act_test, pos_test_x))
    print("Classifier y score:", classifier_y.score(act_test, pos_test_y))
    #plot_pred_pos(pred_x, pred_y, pos_test_x, pos_test_y)

def posfromact(resolution, path, nb_train):
    activities = load_reservoir_states(path)
    positions = load_positions(path)

    # Preprocessing positions : array of (res[0] + res[1]) ints 
    # (0s and two 1s to indicate x and y)
    disc_pos = preprocess_positions(resolution, positions)
    # Train and test splitting
    act_train, pos_train = activities[:nb_train], disc_pos[:nb_train]
    act_test, pos_test = activities[nb_train:], disc_pos[nb_train:]
    # Standardizing activity
    scaler = preprocessing.StandardScaler().fit(act_train)
    act_train = scaler.transform(act_train)
    act_test = scaler.transform(act_test)

    classifier = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(20,20),
                                random_state=1, max_iter=200)
    classifier.fit(act_train, pos_train)

    # Classification
    pos_predicted = classifier.predict(act_test)
    print("Classifier score :", classifier.score(act_test, pos_test))
    plot_comparison_pred_test(resolution, pos_predicted, pos_test)

def position_from_sensors(resolution, path, nb_train):
    sensors = load_sensors(path)
    positions = load_positions(path)

    # Preprocessing positions : array of (res[0] + res[1]) ints 
    # (0s and two 1s to indicate x and y)
    disc_pos = preprocess_positions(resolution, positions)
    # Train and test splitting
    act_train, pos_train = sensors[:nb_train], disc_pos[:nb_train]
    act_test, pos_test = sensors[nb_train:], disc_pos[nb_train:]
    # Standardizing sensors
    scaler = preprocessing.StandardScaler().fit(act_train)
    act_train = scaler.transform(act_train)
    act_test = scaler.transform(act_test)

    classifier = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(50,),
                                random_state=1, max_iter=200)
    classifier.fit(act_train, pos_train)

    # Classification
    pos_predicted = classifier.predict(act_test)
    print("Classifier score :", classifier.score(act_test, pos_test))
    #plot_comparison_pred_test(resolution, pos_predicted, pos_test)

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

def plot_activity_map(neuron, positions, explo_map, resolution):
    map = activity_map(neuron, positions, explo_map, resolution)
    plt.imshow(map.T, cmap = 'bwr', origin='lower', vmin=-1, vmax=1)
    plt.show()

path = "/home/heloise/Mnémosyne/splitter-cells/trials/random_walls/reservoir_states/"
position_from_sensors((6,10), path, 2000)
position_from_activity((6,10), path, 2000)