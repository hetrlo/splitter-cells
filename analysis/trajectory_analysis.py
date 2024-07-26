"""
An analysis of the bot's trajectory, to quantify its exploration of the maze

"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from math import *
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy.ndimage import median_filter

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
    plt.legend(('Xpred', 'Xtest'))
    plt.title('Position X from sensors')
    #plt.plot(np.abs(x_pred-x_test), color='black')
    plt.show()
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    plt.plot(y_pred, color='blue')
    plt.plot(y_test, color='orange', linestyle=":")
    plt.legend(('Ypred', 'Ytest'))
    plt.title('Position Y from sensors')
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
    plt.legend(('Xpred', 'Xtest'))
    plt.title('Position X from activity')
    plt.plot(np.abs(x_pred-x_test), color='black')
    plt.show()
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    plt.plot(y_pred, color='blue')
    plt.plot(y_test, color='orange', linestyle=":")
    plt.title('Position Y from activity')
    plt.plot(np.abs(y_pred-y_test), color='black')
    plt.show()

def position_from_activity(resolution, path, nb_train):
    activities = load_reservoir_states(path)[:-1200]
    positions = load_positions(path)[:-1200]

    # Preprocessing positions : (0s and two 1s to indicate x and y)
    (disc_pos_x, disc_pos_y) = process_positions(resolution, positions)
    # Train and test splitting
    scaler = preprocessing.StandardScaler()
    scaler.fit(activities)
    activities = scaler.transform(activities)
    act_train, pos_train_x, pos_train_y = activities[:nb_train], disc_pos_x[:nb_train], disc_pos_y[:nb_train]
    act_test, pos_test_x, pos_test_y = activities[nb_train:], disc_pos_x[nb_train:], disc_pos_y[nb_train:]

    classifier_x = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(40,20,10),
                                random_state=1, max_iter=500)
    classifier_x.fit(act_train, pos_train_x)

    classifier_y = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(40,20),
                                random_state=1, max_iter=500)
    classifier_y.fit(act_train, pos_train_y)

    # Classification
    pred_x = classifier_x.predict(act_test)
    pred_y = classifier_y.predict(act_test)
    print("Classifier x score:", classifier_x.score(act_test, pos_test_x))
    print("Classifier y score:", classifier_y.score(act_test, pos_test_y))
    plt.title('Position from activity')
    plot_pred_pos(pred_x, pred_y, pos_test_x, pos_test_y)

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
                               hidden_layer_sizes=(40,40),
                                random_state=1, max_iter=1000)
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
    scaler = preprocessing.StandardScaler()
    scaler.fit(act_train)
    act_train = scaler.transform(act_train)
    act_test = scaler.transform(act_test)

    classifier = MLPClassifier(solver='adam', alpha=1e-5, 
                               hidden_layer_sizes=(50,),
                                random_state=1, max_iter=200)
    classifier.fit(act_train, pos_train)

    # Classification
    pos_predicted = classifier.predict(act_test)
    print("Classifier score :", classifier.score(act_test, pos_test))
    plot_comparison_pred_test(resolution, pos_predicted, pos_test)

# Trying to decode position
from sklearn.neural_network import MLPRegressor
def continuous_position_from_activity(path, nb_train):

    # Loading training and testing data and preprocessing
    activities = load_reservoir_states(path)[:-200]
    positions = load_positions(path)[:-200]
    x = np.array([pos[0] for pos in iter(positions)])
    y = np.array([pos[1] for pos in iter(positions)])

    # Standardizing position
    x, y = standardize(x), standardize(y)
    act_train, x_train, y_train = activities[:nb_train], x[:nb_train], y[:nb_train] 
    act_test, x_test, y_test = activities[nb_train:], x[nb_train:], y[nb_train:]

    # Definition of the regressor
    regressor_x = MLPRegressor(random_state=1, hidden_layer_sizes=(40, 20), max_iter=1000, solver='sgd').fit(act_train, x_train)
    regressor_y = MLPRegressor(random_state=1, hidden_layer_sizes=(40, 20), max_iter=1000, solver='sgd').fit(act_train, y_train)

    # Regression
    x_predicted = regressor_x.predict(act_test)
    y_predicted = regressor_y.predict(act_test)
    print("Classifier score x:", regressor_x.score(act_test, x_test))
    print("Classifier score y:", regressor_y.score(act_test, y_test))

    #pos_predicted = median_filter(pos_predicted, 10)
    print([(x_predicted[i], y_predicted[i]) for i in range(10)])
    print([(x_test[i],y_test[i]) for i in range(10)])
    time = np.linspace(0,len(x_test), len(x_test))
    plt.plot(time, x_predicted, y_predicted, color='red')
    plt.plot(time, x_test, y_test, color='blue')
    plt.show()

# Verif method
def continuous_orientation_from_activity(path, nb_train):
    # Loading training and testing data and preprocessing
    activities = load_reservoir_states(path)
    orientations = load_orientations(path)
    # Standardizing sensors
    orientations = standardize(orientations)
    act_train, ori_train = activities[:nb_train], orientations[:nb_train]
    act_test, ori_test = activities[nb_train:], orientations[nb_train:]

    # Definition of the regressor
    regressor = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(20, 10)).fit(act_train, ori_train)

    # Regression
    ori_predicted = regressor.predict(act_test)
    print("Classifier score :", regressor.score(act_test, ori_test))
    print(ori_predicted[:10])
    print(ori_test[:10])
    plt.plot(ori_predicted[:-100], color='red')
    plt.plot(ori_test[:-100], color='blue')
    plt.show()

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

path = "/home/heloise/Mnémosyne/splitter-cells-results/traj/esn/RR-LL/"

position_from_activity((15,25), path, 2000)