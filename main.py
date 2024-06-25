""" Main script to run.
Before running the script, certain configurations are required:

- task :
        1) 'R-L' (alternation task)
        2) 'RR-LL' (half-alternation task)

- simulation_mode:
                1) 'walls': the bot navigates and takes direction automatically using Braitenberg algorithms.
                            Walls are added to guide the bot in the right direction.
                   Some walls are added so as to force the bot taking the right direction
                2) 'data': the bot is data-driven and navigates based on the provided position file.
                3) 'esn': the bot moves based on ESN predictions, trained using supervised learning.

- save_reservoir_states: set to True if the reservoir states and the bot's positions and orientation need to be recorded
- save_bot_states: set to True if the bot's positions and orientation need to be recorded. Orientation contains the 
                   orientation of the bot, output its position, and input is a concatenation of cues (if there are) and
                   of the sensors values
- path_to_save: folder to save

"""



import matplotlib.animation as animation
from experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np

maze = 'maze_other' # 'maze', 'maze_four', 'random_walls', 'maze_other'
task = 'wander' #'RR-LL', 'R-L', 'wander'
simulation_mode = 'esn'  # 'data', 'walls', 'esn'
cues = False
save_reservoir_states = True
save_bot_states = True
path_to_save = './trials/trained_maze_other__new_maze/'


if __name__ == '__main__':
    if task == 'R-L':
        print('Run the alternation task (R-L) ...')
        if cues:
            model_file = "model_settings/model_RL_cues_.json"
            data_folder = "data/R-L_60/cues/"
        else:
            model_file = "model_settings/model_RL_no_cues_.json"
            data_folder = "data/R-L_60/no_cues/"
    elif task == 'RR-LL':
        print('Run the half-alternation task (RR-LL) ...')
        if cues:
            model_file = "model_settings/model_RR-LL_cues.json"
            data_folder = "data/RR-LL/cues/"
        else:
            model_file = "model_settings/model_RR-LL_no_cues.json"
            data_folder = "data/RR-LL/no_cues/"
            data_folder = "data/RR-LL/no_cues/error_case/"
    if maze == 'random_walls':
        print("Run the wander around and find out task :)")
        model_file = "trials/training_random_walls/model_settings_wander.json"
        data_folder = "trials/training_random_walls/"
    elif maze == 'maze_other':
        print("Wandering around in the 'maze_other' maze")
        model_file = "trials/training_maze_other/model_settings_maze_other.json"
        data_folder = "trials/training_maze_other/"

    else:
        raise Exception("Task name {}".format(task) + " is not recognized.")

    # Set up the experiment
    exp = Experiment(model_file, data_folder, simulation_mode, maze, task, cues,
                     save_reservoir_states=save_reservoir_states,
                     save_bot_states=save_bot_states)

    # Set up the animation
    anim = animation.FuncAnimation(exp.simulation_visualizer.fig, exp.run,
                                   frames=50000, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()

    # Save data
    if save_bot_states:
        np.save(path_to_save + 'positions.npy', exp.bot.all_positions)
        np.save(path_to_save + 'output.npy', exp.bot.all_orientations)

        if cues:
            # concatenate sensors with cues
            input = np.concatenate((exp.bot.all_sensors_vals,  exp.bot.all_cues), axis=1)
            np.save(path_to_save + 'input.npy', input)
        else:
            np.save(path_to_save + 'input.npy', exp.bot.all_sensors_vals)

    if save_reservoir_states:
        np.save(path_to_save + 'reservoir_states.npy', exp.model.reservoir_states)


