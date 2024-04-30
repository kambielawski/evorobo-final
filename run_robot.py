from robot import SpringRobot
from simulation import Simulation

constraints = {
    'min_spring_length': 1,
    'max_spring_length': 2,
    'n_points': 6
}

robot = SpringRobot(constraints)

sim = Simulation(robot)
sim.run_simulation()
sim.draw(0)
sim.optimize_brain(15)
sim.draw(sim.max_steps)
sim.make_and_watch_movie('movie.mp4')
