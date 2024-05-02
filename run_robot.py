import pickle

from robot import SpringRobot
from simulation import Simulation


##### FROM SCRATCH #####
# constraints = {
#     'min_spring_length': 1,
#     'max_spring_length': 2,
#     'n_points': 5
# }

# robot = SpringRobot(constraints)

# sim = Simulation(robot)
# print(sim.hidden_neuron_bias)
# sim.run_simulation()
# sim.draw(0)
# sim.optimize_brain(20)
# # sim.save_robot('robot.pkl')

# sim.run_simulation()
# print(sim.loss)
# sim.run_simulation()
# print(sim.loss)
# sim.draw(sim.max_steps)
# sim.make_and_watch_movie('movie.mp4')


##### FROM PICKLE #####
with open('evo_run.pkl', 'rb') as pf:
    evo_run = pickle.load(pf)

robot = evo_run.best_robot_over_generations[-1]
print(robot.fitness)
print(robot.losses)
# print(robot.brain.hidden_neuron_bias)
sim = Simulation(robot)
sim.run_simulation()
print(sim.loss[None])
sim.draw(0)
sim.make_and_watch_movie('best.mp4')