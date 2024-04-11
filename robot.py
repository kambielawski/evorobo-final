

class Robot:
    '''
    This class handles the inner-loop optimization and evolutionary
    processes of a single individual.
    '''
    def __init__(self):
        self.fitness = 0
        self.genome = []   # A set of (x,y) points
        self.body = None   # A data structure interpretable by taichi

        self.initialize_genome()


    def run(self):
        '''
        Call functions from simulation.py to run the genome.
        '''
        self.build_body()
        self.fitness = simulate(self.body)


    def initialize_genome(self):
        '''
        Randomly initialize a set of (x,y) points that make 
        up the body of the robot.
        '''
        pass


    def build_body(self):
        '''
        Take the genome and connect the set of (x,y) with springs,
        using particular body rules.
        '''
        pass


    def mutate(self):
        '''
        Mutate the genome by randomly changing the robot's 
        morphology. 
        p_add - probability of adding a point
        p_remove - probability of removing a point
        '''
        pass
