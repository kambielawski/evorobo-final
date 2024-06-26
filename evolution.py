import time
import pickle
import numpy as np
import taichi as ti
from robot import SpringRobot
from simulation import Simulation

class Evolution:
    def __init__(self, experiment_parameters):
        self.child_population = []
        self.parent_population = []
        self.current_generation = 0
        self.best_robot_over_generations = []

        self.population_size = experiment_parameters['population_size']
        self.max_generations = experiment_parameters['max_generations']
        self.inner_iterations = experiment_parameters['inner_iterations']
        self.robot_constraints = experiment_parameters['robot_constraints']

    def initialize_population(self):
        self.child_population = [SpringRobot(self.robot_constraints) for _ in range(self.population_size)]

    def evolve(self):
        self.initialize_population()
        while self.current_generation < self.max_generations:
            start = time.time()
            print('Generation: ', self.current_generation + 1)
            self.run_one_generation()
            print('Time taken: ', time.time() - start)
            self.current_generation += 1

    def run_one_generation(self):
        '''
        Simulate each individual
        '''
        # Simulate and evaluate child robots
        for robot in self.child_population:
            self.run(robot)
            robot.set_simulated(True)

        # Select: moves subset of children to parent population
        self.select()

        # Mutate: generate new children from selected parents
        self.mutate()

    def run(self, robot):
        sim = Simulation(robot)
        sim.optimize_brain(self.inner_iterations)
        robot.losses = sim.losses

        if np.isnan(robot.losses[-1]):
            robot.fitness = robot.losses[-1]
        else:
            robot.fitness = robot.losses[0] - robot.losses[-1]

        print('Losses: ', robot.losses)
        print('Fitness: ', robot.fitness)

        if self.current_generation == 0 or self.current_generation == self.max_generations - 1:
            sim.draw(0)
            sim.make_and_watch_movie(f'movie_{self.current_generation}.mp4')

    def mutate(self):
        self.child_population = []
        for robot in self.parent_population:
            child = robot.create_offspring()
            self.child_population.append(child)

    def get_best_robot(self): 
        # Filter out nan fitnesses from parent population and select min
        if all([np.isnan(robot.fitness) for robot in self.parent_population]):
            best_robot = self.parent_population[0]
        else:
            best_robot = min([robot for robot in self.parent_population if not np.isnan(robot.fitness)])
        return best_robot


    def select(self):
        '''
        Parallel hillclimbers
        '''
        if len(self.parent_population) == 0:
            self.parent_population = self.child_population
            best_robot = self.get_best_robot()
            self.best_robot_over_generations.append(best_robot)
            return
        
        for i, robot in enumerate(self.child_population):
            if np.isnan(self.parent_population[i].fitness):
                self.parent_population[i] = robot
            elif robot.fitness > self.parent_population[i].fitness:
                self.parent_population[i] = robot

        best_robot = self.get_best_robot()
        self.best_robot_over_generations.append(best_robot)

    def pickle_evo(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
