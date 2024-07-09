import reservoirpy as rpy
rpy.verbosity(0)
from reservoirpy.nodes import Reservoir
import os, json
import numpy as np

class Pool:
    def __init__(self, model_file, data_setup, save_reservoir_states):

        self.model_file = model_file
        self.save_reservoir_states = save_reservoir_states
        self.data_setup = data_setup

        with open(os.path.join(os.path.dirname(__file__), model_file)) as f:
            _ = json.load(f)

        self.cues = bool(_['cues'])
        
        units = _['n_units']
        input_scaling = _['input_scaling']
        leak_rate = _['leak_rate']
        spectral_radius = _['spectral_radius']
        input_connectivity = _['input_connectivity']
        connectivity = _['connectivity']
        noise_rc = _['noise_rc']
        seed = _['seed']
        nb_train = _['nb_train']
        warmup = _['warmup']

        self.reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                                  lr=leak_rate, rc_connectivity=connectivity,
                                  input_connectivity=input_connectivity, seed=seed, noise_rc=noise_rc)

        self.save_reservoir_states = save_reservoir_states
        self.reservoir_states = []
        setup = np.load(data_setup + "input.npy")
        self.reservoir.run(setup[:nb_train+warmup]) # initialize the reservoir

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

        # Updating the reservoir
        self.reservoir(input)
        # Saving the activity
        if self.save_reservoir_states:
            self.record_states()