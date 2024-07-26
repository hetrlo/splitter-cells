import reservoirpy as rpy
rpy.verbosity(0)
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS
from reservoirpy.observables import nrmse, rsquare
import os, json
import numpy as np
import sys
sys.path.append('/home/heloise/Mnémosyne/splitter-cells/')
print(sys.path)



def split_train_test(input, output, nb_train, max_len_test=100000):
    """ Splits the input and output lists in two parts: one list for the training phase, one for the testing phase.
    Inputs:
    - input: input list
    - output: output list
    - nb_train:number of element in the training lists. nb_test = len(input)-nb_train
    - max_len_test : max length of the testing list (so it is not too big).
    Outputs:
    - X_train, X_test: two lists from input that were split at the nb_train index. len(X_test) <= max_len_test
    - Y_train, Y_test: two lists from output that were split at the nb_train index. len(Y_test) <= max_len_test"""

    X_train, Y_train, X_test, Y_test = input[:nb_train], output[:nb_train], input[nb_train:], output[nb_train:]
    if len(X_test) > max_len_test:
        X_test, Y_test = X_test[:max_len_test], Y_test[:max_len_test]

    return X_train, Y_train, X_test, Y_test


class Model:
    def __init__(self, model_file, data_folder, save_reservoir_states):

        self.model_file = model_file
        self.data_folder = data_folder
        self.save_reservoir_states = save_reservoir_states

        with open(os.path.join(os.path.dirname(__file__), model_file)) as f:
            _ = json.load(f)

        self.nb_train = _['nb_train']
        self.cues = bool(_['cues'])

        units = _['n_units']
        input_scaling = _['input_scaling']
        leak_rate = _['leak_rate']
        spectral_radius = _['spectral_radius']
        regularization = _['regularization']
        input_connectivity = _['input_connectivity']
        connectivity = _['connectivity']
        noise_rc = _['noise_rc']
        warmup = _['warmup']
        seed = _['seed']

        self.input = np.load(data_folder + 'input.npy')
        output = np.load(data_folder + 'output.npy')
        output = output.reshape(len(output), 1)
        self.positions = np.load(data_folder + 'positions.npy')

        if self.cues:
            print('Fit model with sensors and contextual cues as input...')
        else:
            print('Fit model with sensors as input...')

        self.reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                                  lr=leak_rate, rc_connectivity=connectivity,
                                  input_connectivity=input_connectivity, seed=seed, noise_rc=noise_rc)
        
        readout = Ridge(1, ridge=regularization)

        # To add feedback
        #self.reservoir <<= readout

        self.esn = self.reservoir >> readout
        X_train, Y_train, X_test, Y_test = split_train_test(self.input, output, self.nb_train)

        #self.esn.fit(X_train, Y_train, warmup=warmup, force_teachers=True)
        self.esn.fit(X_train, Y_train, warmup=warmup)
        self.save_reservoir_states = save_reservoir_states
        self.reservoir_states = []

        # Learning position
        readout_position = Ridge()
        self.esn_pos = self.reservoir >> readout_position

        '''def euclidian_dist(p1, p2):
            return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        
        def standardize(array):
            array = np.abs(array)
            max_arr = np.max(array)
            min_arr = np.min(array)
            if max_arr==0:
                return array
            if min_arr == max_arr:
                return array / max_arr
            return (array - min_arr) / (max_arr - min_arr)
        
        def dist_point(positions, point):
            dist = np.array([euclidian_dist(pos, point) for pos in iter(positions)])
            dist = standardize(dist)
            return 1 - dist

        X_train_pos, Y_train_pos, X_test_pos, Y_test_pos = split_train_test(self.input, self.positions, self.nb_train)
        self.esn_pos.fit(X_train_pos, Y_train_pos, warmup=warmup)
        self.decoded_pos = []'''
        

    def record_states(self):
        """ Function that records the reservoir state at the given position in the maze.
        Inputs:
        - bot_position: current position of the bot
        - reservoir: reservoir model
        if self.where == None: record the reservoir state everywhere in the maze
        else: record the reservoir state only in the last corridor just before the decision point.
        """
        s = []
        for val in np.array(self.reservoir.state()[0]):
            s.append(val)
        self.reservoir_states.append(s)

    def process(self, sensors, cues=None):
        if self.cues:
            input = np.concatenate((sensors['value'].ravel(), np.array(cues))).reshape(1, 10)
        else:
            input = np.array(sensors['value']).reshape(1, 8)

        output = np.array(self.esn(input))[0][0]
        if self.save_reservoir_states:
            self.record_states()
        
        '''decoded_position = np.array(self.esn_pos(input))
        if self.save_reservoir_states:
            self.decoded_pos.append(decoded_position)'''

        return output