import os
import numpy as np

# from simulation import draw, make_and_watch_movie, initialize_body, initialize_brain, run_simulation, optimize_brain

# from simulation import Simulation

ACTUATION_INIT_FREQ = 0.35

class Brain:
    def __init__(self, hidden_neuron_weights=None, motor_neuron_weights=None, hidden_neuron_bias=None, motor_neuron_bias=None): 
        self.hidden_neuron_weights = hidden_neuron_weights
        self.motor_neuron_weights = motor_neuron_weights
        self.hidden_neuron_bias = hidden_neuron_bias
        self.motor_neuron_bias = motor_neuron_bias

class SpringRobot:
    '''
    This class handles the inner-loop optimization and evolutionary
    processes of a single individual.
    '''
    def __init__(self, constraints):
        self.fitness = 0
        self.body_points = None   # A data structure interpretable by taichi
        self.simulated = False
        self.brain = Brain()

        self.constraints = constraints
        self.min_spring_length = constraints['min_spring_length']
        self.max_spring_length = constraints['max_spring_length']
        self.n_points = constraints['n_points']

        # self.generate_random_body_points(self.n_points) # Initialize random genome
        # self.generate_box_body_points()
        self.generate_pentagon_body_points()
        # self.generate_spring_body_points()

    # def run(self):
    #     '''
    #     Call functions from simulation.py to run the genome.
    #     '''

    def generate_valid_point(self, points):
        '''
        Generate a valid point within the constraints.
        '''
        valid = False
        attempts = 0
        
        while not valid and attempts < 100:  # Prevent infinite loops
            # Choose a random existing point
            base_point = points[np.random.randint(0, len(points))]
            
            # Generate a random angle in radians
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Generate a random distance between a and b
            distance = np.random.uniform(self.min_spring_length, self.max_spring_length)
            
            # Calculate the new point's coordinates
            new_x = base_point[0] + distance * np.cos(angle)
            new_y = base_point[1] + distance * np.sin(angle)
            
            # Check if the new point is valid (not too close to any existing point)
            new_point = np.array([new_x, new_y])
            distances = np.linalg.norm(points - new_point, axis=1)
            if np.all(distances >= self.min_spring_length):
                valid = True
            
            attempts += 1

        return new_point

    def generate_random_body_points(self, N):
        """
        Take the genome and connect the set of (x,y) with springs,
        using particular body rules.
        
        Args:
        N (int): Number of points to generate.
        a (float): Minimum distance from an existing point.
        b (float): Maximum distance from an existing point.
        
        Returns:
        np.array: Array of shape (N, 2) containing the generated points.
        """
        points = np.array([[0,0]]) # Initialize the array with the first point at (0, 0)

        for _ in range(1, N):
            new_point = self.generate_valid_point(points)
            # Extend the array with the new point
            points = np.concatenate([points, [new_point]])
        
        self.body_points = points

    def generate_box_body_points(self):
        '''
        Generate a box-shaped body
        '''
        points = np.array([[0,0], [1,0], [1,1], [0,1]])
        self.body_points = points.astype(np.float32)

    def generate_pentagon_body_points(self):
        '''
        Generate a pentagon-shaped body
        '''
        points = np.array([
            [0, 1.0],    # Vertex 1 (Approximately (0, 1))
            [0.9510565162951535, 0.3090169943749474],  # Vertex 2
            [0.5877852522924731, -0.8090169943749475], # Vertex 3
            [-0.587785252292473, -0.8090169943749475], # Vertex 4
            [-0.9510565162951536, 0.3090169943749473]  # Vertex 5
        ])
        self.body_points = points.astype(np.float32)

    
    def get_sim_body_points(self, x_start, y_start, body_size):
        self.sim_body_points = self.body_points.copy()

        # 1. Move body into +X and +Y plane
        min_x = min(self.sim_body_points[:, 0])
        min_y = min(self.sim_body_points[:, 1])
        if min_x < 0: 
            self.sim_body_points[:, 0] += -min_x
        if min_y < 0:
            self.sim_body_points[:, 1] += -min_y

        # 2. Scale down the points to be within body_size
        max_x = max(self.sim_body_points[:, 0])
        max_y = max(self.sim_body_points[:, 1])
        scale_factor_x = max_x / body_size
        scale_factor_y = max_y / body_size
        self.sim_body_points[:, 0] /= scale_factor_x
        self.sim_body_points[:, 1] /= scale_factor_y
        
        # 3. Add x_start, y_start
        self.sim_body_points[:, 0] += x_start
        self.sim_body_points[:, 1] += y_start

        return self.sim_body_points

    def generate_body_springs(self, body_points):
        def make_spring(a,b, motor=False):
            resting_length = np.sqrt((body_points[a][0] - body_points[b][0])**2 + (body_points[a][1] - body_points[b][1])**2)
            return [a,b, resting_length, motor, ACTUATION_INIT_FREQ]

        n_points = len(body_points)
        springs = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                springs.append(make_spring(i,j, motor=True))

        return springs
    
    def mutate(self):
        '''
        Mutate the genome by randomly changing the robot's 
        morphology. 
        p_add - probability of adding a point
        p_remove - probability of removing a point
        '''
        # Get the index of the extreme points
        rightmost_idx = np.argmax(self.body_points[:, 0])
        leftmost_idx = np.argmin(self.body_points[:, 0])
        topmost_idx = np.argmax(self.body_points[:, 1])
        bottommost_idx = np.argmin(self.body_points[:, 1])

        # Randomly delete one of the extreme points
        rand = np.random.rand()
        if 0 <= rand <= 0.25:
            self.body_points = np.delete(self.body_points, rightmost_idx, axis=0)
        elif 0.25 < rand <= 0.5:
            self.body_points = np.delete(self.body_points, leftmost_idx, axis=0)
        elif 0.5 < rand <= 0.75:
            self.body_points = np.delete(self.body_points, topmost_idx, axis=0)
        else:
            self.body_points = np.delete(self.body_points, bottommost_idx, axis=0)

        # Add a new point randomly
        new_point = self.generate_valid_point(self.body_points)
        self.body_points = np.concatenate([self.body_points, [new_point]])

    def create_offspring(self):
        '''
        Creates a child robot
        '''
        child = SpringRobot(self.constraints)
        child.body_points = self.body_points.copy() # Copy the genome
        child.mutate()
        child.set_simulated(False)
        return child
    
    def set_simulated(self, simulated):
        self.simulated = simulated

    def draw(self, frame_offset):
        '''
        Draw the robot
        '''
        draw(frame_offset)

    def make_video(self, file_name):
        '''
        Create a video of the robot simulation
        '''
        make_and_watch_movie(file_name)

