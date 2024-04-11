from robot import Robot

class Evolution:
    def __init__(self, experiment_parameters):
        self.child_population = []
        self.parent_population = []
        self.current_generation = 0

        self.population_size = experiment_parameters['population_size']
        self.max_generations = experiment_parameters['max_generations']

    def initialize_population(self):
        self.population = [Robot() for _ in range(self.population_size)]

    def evolve(self):
        self.initialize_population()
        while self.current_population < self.max_generations:
            self.run_one_generation()

    def run_one_generation(self):
        '''
        Simulate each individual
        '''
        # Simulate and evaluate child robots
        for robot in self.child_population:
            robot.run()

        # Select: moves subset of children to parent population
        self.select()

        # Mutate: generate new children from selected parents

    def select(self):
        '''
        Tournament selection? 
        '''
        pass 

        
        
        
