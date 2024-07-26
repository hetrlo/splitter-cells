from maze import Maze, MazeFour, RandomWalls, MazeOther, Empty
from bot import Bot
from simulation_visualizer import SimulationVisualizer
import numpy as np
from esn_model import Model
from neuron_pool import Pool
from typing import *

"""
- automatic navigation with walls (braitenberg+walls) 'walls_directed'
- navigation with data (controlled by generated data) 'data_directed'
- navigation thanks to the models (use ESN) 'esn_directed' 


- input type: cues, no_cues

"""



class Experiment:
    """
    This class runs the experiment.
    """
    def __init__(self, model_file, data_folder, data_setup : Optional[str], simulation_mode, maze, task, cues, noise, save_reservoir_states, save_bot_states):

        self.task = task
        self.simulation_mode = simulation_mode
        self.cues = cues

        if task == "R-L":
            sensor_size = 60
        # Bigger sensor because the task is more challenging
        elif task == "RR-LL":
            sensor_size = 80
        elif task == "wander":
            sensor_size = 80

        self.bot = Bot(noise, save_bot_states, sensor_size=sensor_size)
        # Init of the maze
        if maze == 'maze':
            self.maze = Maze(simulation_mode=simulation_mode)
        elif maze == 'maze_four':       
            self.maze = MazeFour(simulation_mode=simulation_mode)
        elif maze == 'random_walls':
            self.maze = RandomWalls(simulation_mode=simulation_mode)
        elif maze == 'maze_other':
            self.maze = MazeOther(simulation_mode=simulation_mode)
        elif maze == 'empty':
            self.maze = Empty(simulation_mode=simulation_mode)
        self.simulation_visualizer = SimulationVisualizer(self.bot.n_sensors)
        self.model_file = model_file
        self.data_folder = data_folder
        self.data_setup = data_setup
        self.save_reservoir_states = save_reservoir_states

        if self.simulation_mode == 'data' or self.simulation_mode == 'mix':
            self.output = np.load(self.data_folder + 'output.npy')
            self.positions = np.load(self.data_folder + 'positions.npy')
            self.input = np.load(self.data_folder + 'input.npy')

        elif self.simulation_mode == 'esn':
            self.model = Model(model_file=self.model_file, data_folder=self.data_folder,
                               save_reservoir_states=self.save_reservoir_states)
            self.bot.position = self.model.positions[self.model.nb_train-1]
            # To choose where the bot starts after training
            #self.bot.position = [50,100]

        if self.simulation_mode == 'mix':
            self.pool = Pool(model_file=self.model_file, data_setup=self.data_setup, 
                               save_reservoir_states=self.save_reservoir_states)

        self.maze.draw(self.simulation_visualizer.ax, grid=True, margin=15)
        self.bot.draw(self.simulation_visualizer.ax)

    def run(self, frame):
        if self.simulation_mode == 'data' or self.simulation_mode == 'mix':
            self.bot.orientation = self.output[frame]
            self.bot.position = self.positions[frame]
            self.bot.update(self.maze, cues=self.cues)

            if self.simulation_mode =='mix':
                self.pool.process(self.bot.sensors, self.cues)

        else:
            if self.cues:
                cues = self.bot.update_cues(self.task)
            else:
                cues = None

            if self.simulation_mode == 'walls':
                self.bot.update_position(self.maze)
                self.bot.compute_orientation()
                if self.task == 'R-L':
                    self.maze.update_walls(self.bot.position)
                elif self.task == 'RR-LL':
                    self.maze.update_walls_RR_LL(self.bot.position)
                elif self.task == 'wander':
                    pass

            elif self.simulation_mode == 'esn':
                self.bot.update_position(self.maze)
                self.bot.orientation = self.model.process(self.bot.sensors, cues)                

            self.bot.update(self.maze, cues=self.cues)

        self.simulation_visualizer.update_plot(frame, self.bot.position, self.bot.sensors['value'])

        return self.bot.artists, self.simulation_visualizer.trace, self.simulation_visualizer.plots





















