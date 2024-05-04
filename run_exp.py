import argparse
import time

from evolution import Evolution
from robot import SpringRobot
from simulation import Simulation

'''
parser = argparse.ArgumentParser()
parser.add_argument('--exp_file', type=str, required=True)

args = parser.parse_args()

exp_file = args.exp_file
exp_params = eval(open(exp_file).read())

exp = Evolution(exp_params)
exp.evolve()
exp.pickle_evo()

'''

exp_params = {
    'population_size': 1,
    'max_generations': 50,
    'inner_iterations': 15,
    'robot_constraints': {
        'min_spring_length': 1,
        'max_spring_length': 2,
        'n_points': 5
    }
}

########

start = time.time()

evo_run = Evolution(exp_params)
evo_run.evolve()
evo_run.pickle_evo('evo_run.pkl')

total_time = time.time() - start
print('Experiment Time: ', total_time)
print('Avg. time per generation: ', total_time / exp_params['max_generations'])

# print(evo_run.best_robot_over_generations)