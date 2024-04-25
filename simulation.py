import os
import math
import taichi as ti
import numpy as np
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt

from robot import SpringRobot

# Initialization
ti.init(arch=ti.metal, default_fp = ti.f32)

ground_height = 0.1
dt = 0.02
gravity = -9.8
sim_damping = 1.0
spring_stiffness = 100.0
# n_points = 4
ITERS = 10
LEARNING_RATE = 0.01
N_HIDDEN_NEURONS = 32
N_SIN_WAVES = 10
GOAL_POSITION = [0.9,0.2]


@ti.data_oriented
class Simulation:
    def __init__(self, robot):
        self.robot_starting_points = robot.get_sim_body_points(x_start=0.2, y_start=0.2, body_size=0.2)
        self.springs = robot.generate_body_springs(self.robot_starting_points)
        self.n_points = len(self.robot_starting_points)
        self.n_springs = len(self.springs)
        self.max_steps = 100

        self.initialize_fields()
        self.initialize_body()
        self.initialize_brain()

    def n_sensors(self):
        # n_sin_waves simulates central pattern generators
        # each object has 4 internal sensors
        # 2 global sensors... 
        return N_SIN_WAVES + 4*self.n_points + 2

    def initialize_fields(self):
        self.spring_anchor_a = ti.field(ti.i32)
        self.spring_anchor_b = ti.field(ti.i32)
        self.spring_resting_lengths = ti.field(ti.f32)
        self.spring_actuation = ti.field(ti.i32)
        self.spring_frequency = ti.field(ti.f32)

        vec = lambda: ti.Vector.field(2, dtype=ti.f32)
        self.positions = vec()
        self.velocities = vec()
        self.spring_restoring_forces = vec()
        self.spring_forces_on_objects = vec()
        self.center = vec() # Center of the robot
        self.goal = vec()

        self.hidden_neuron_values = ti.field(ti.f32)
        self.motor_neuron_values = ti.field(ti.f32)
        self.hidden_neuron_bias = ti.field(ti.f32)
        self.motor_neuron_bias = ti.field(ti.f32)
        self.weights_sh = ti.field(ti.f32)
        self.weights_hm = ti.field(ti.f32)

        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.positions)
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.positions.grad)
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.velocities)
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.velocities.grad)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_anchor_a, self.spring_anchor_b, self.spring_resting_lengths)
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_springs).place(self.spring_restoring_forces)
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.spring_forces_on_objects)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_actuation)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_frequency)
        ti.root.dense(ti.i, self.max_steps).place(self.center)
        ti.root.place(self.goal)

        ti.root.dense(ti.ij, (self.max_steps, N_HIDDEN_NEURONS)).place(self.hidden_neuron_values)
        ti.root.dense(ti.ij, (self.max_steps, self.n_springs)).place(self.motor_neuron_values)
        ti.root.dense(ti.i, N_HIDDEN_NEURONS).place(self.hidden_neuron_bias)
        ti.root.dense(ti.i, self.n_springs).place(self.motor_neuron_bias)

        ti.root.dense(ti.ij, (N_HIDDEN_NEURONS, self.n_sensors())).place(self.weights_sh)
        ti.root.dense(ti.ij, (self.n_springs, N_HIDDEN_NEURONS)).place(self.weights_hm)

        self.loss = ti.field(dtype=ti.f32, shape=()) # 0-dimensional tensor (i.e. fp scalar)

        ti.root.lazy_grad()

    def initialize_body(self):
        self.goal[None] = GOAL_POSITION

        # Create points in space and initialize w/ zero velocity
        for object_idx in range(self.n_points):
            self.positions[0, object_idx] = self.robot_starting_points[object_idx]
            self.velocities[0, object_idx] = [0,0]

        # Create springs to connect the points in space
        for spring_idx in range(self.n_springs):
            s = self.springs[spring_idx]
            self.spring_anchor_a[spring_idx] = s[0]
            self.spring_anchor_b[spring_idx] = s[1]
            self.spring_resting_lengths[spring_idx] = s[2]
            self.spring_actuation[spring_idx] = s[3]
            self.spring_frequency[spring_idx] = s[4]

        # Initialize the rest of the positions over time to zero
        for i in range(1,self.max_steps):
            for j in range(self.n_points):
                self.positions[i,j][0] = 0.0
                self.positions[i,j][1] = 0.0
                self.velocities[i,j][0] = 0.0
                self.velocities[i,j][1] = 0.0

        for i in range(self.max_steps):
            for j in range(self.n_springs):
                self.spring_restoring_forces[i,j][0] = 0.0
                self.spring_restoring_forces[i,j][1] = 0.0
                # spring_actuation[i,j] = 0.0

        for i in range(self.max_steps):
            for j in range(self.n_points):
                self.spring_forces_on_objects[i,j][0] = 0.0
                self.spring_forces_on_objects[i,j][1] = 0.0

        for i in range(self.max_steps):
            for j in range(N_HIDDEN_NEURONS):
                self.hidden_neuron_values[i,j] = 0.0

    def initialize_brain(self):
        for h in range(N_HIDDEN_NEURONS):
            self.hidden_neuron_bias[h] = (np.random.randn() * 2 - 1) * 0.02
            for s in range(self.n_sensors()):
                self.weights_sh[h, s] = (np.random.randn() * 2 - 1) * 0.02

        for s in range(self.n_springs):
            self.motor_neuron_bias[s] = (np.random.randn() * 2 - 1) * 0.02
            for h in range(N_HIDDEN_NEURONS):
                self.weights_hm[s, h] = (np.random.randn() * 2 - 1) * 0.02

    @ti.kernel
    def compute_loss(self):
        self.loss[None] -= self.positions[self.max_steps-1, 1][0]

    @ti.kernel
    def simulate_springs(self, timestep: ti.i32):
        for spring_idx in range(self.n_springs):
            object_a_idx = self.spring_anchor_a[spring_idx]
            object_b_idx = self.spring_anchor_b[spring_idx]
            position_a = self.positions[timestep-1, object_a_idx]
            position_b = self.positions[timestep-1, object_b_idx]
            dist = position_a - position_b
            length = dist.norm()
            spring_resting_length = self.spring_resting_lengths[spring_idx]

            # Actuation: change the resting state of the spring to oscillate (open-loop)
            spring_resting_length = spring_resting_length + \
                                    self.spring_actuation[spring_idx] * \
                                    self.motor_neuron_values[timestep, spring_idx]
                                    # 0.05*ti.sin(spring_frequency[spring_idx] * timestep*1.0)

            spring_unhappiness = 2*(length - spring_resting_length)
            self.spring_restoring_forces[timestep, spring_idx] = (dt * spring_stiffness * spring_unhappiness / length) * dist

            self.spring_forces_on_objects[timestep, object_a_idx] += -0.35*self.spring_restoring_forces[timestep, spring_idx]
            self.spring_forces_on_objects[timestep, object_b_idx] += 0.35*self.spring_restoring_forces[timestep, spring_idx]
        
    @ti.kernel
    def simulate_objects(self, timestep: ti.i32):
        for object_idx in range(self.n_points):
            old_position = self.positions[timestep-1, object_idx]
            old_velocity = (1-sim_damping) * self.velocities[timestep-1, object_idx] \
            + dt*gravity * ti.Vector([0,1]) \
            + self.spring_forces_on_objects[timestep, object_idx]

            # Collision detection and resolution
            if old_position[1] < ground_height: # and old_velocity[1] < 0:
                # new_position = ti.Vector([old_position[0], ground_height])
                old_velocity = ti.Vector([0,0])

            new_position = old_position + dt * old_velocity
            new_velocity = old_velocity

            self.positions[timestep, object_idx] = new_position
            self.velocities[timestep, object_idx] = new_velocity

    @ti.kernel
    def simulate_neural_network_sh(self, timestep: ti.i32):
        for h in range(N_HIDDEN_NEURONS):
            activation = 0.0
            # Central pattern generators
            for s in ti.static(range(N_SIN_WAVES)):
                activation += self.weights_sh[h, s] * ti.sin(50*timestep * dt + \
                                                        2*math.pi / N_SIN_WAVES * s)
            # 4 sensors for each object
            for j in ti.static(range(self.n_points)):
                offset = self.positions[timestep, j] - self.center[timestep]
                activation += 0.1*self.weights_sh[h, N_SIN_WAVES + 4*j] * offset[0]     # relative x coordinate
                activation += 0.1*self.weights_sh[h, N_SIN_WAVES + 4*j + 1] * offset[1] # relative y coordinate
                activation += 0.1*self.weights_sh[h, N_SIN_WAVES + 4*j + 2] * self.positions[timestep, j][0]
                activation += 0.1*self.weights_sh[h, N_SIN_WAVES + 4*j + 3] * self.positions[timestep, j][1]

            activation += self.weights_sh[h, self.n_points*4 + N_SIN_WAVES]     * \
                                                        (self.goal[None][0] - self.center[timestep][0])
            activation += self.weights_sh[h, self.n_points*4 + N_SIN_WAVES + 1] * \
                                                        (self.goal[None][1] - self.center[timestep][1])

            activation = ti.tanh(activation + self.hidden_neuron_bias[h])
            self.hidden_neuron_values[timestep, h] = activation    

    @ti.kernel
    def simulate_neural_network_hm(self, timestep : ti.i32):
        for spring_idx in range(self.n_springs):
            activation = 0.0
            for h in ti.static(range(N_HIDDEN_NEURONS)):
                activation += self.weights_hm[spring_idx, h] * self.hidden_neuron_values[timestep, h]

            activation = ti.tanh(activation) # + motor_neuron_bias[spring_idx])
            self.motor_neuron_values[timestep, spring_idx] = activation

    @ti.kernel
    def compute_center(self, timestep : ti.i32):
        for object_idx in range(self.n_points):
            self.center[timestep] += self.positions[timestep, object_idx] / self.n_points

    # Simulate
    def step_once(self, timestep : ti.i32):

        self.compute_center(timestep)

        self.simulate_neural_network_sh(timestep)
        self.simulate_neural_network_hm(timestep)

        self.simulate_springs(timestep)

        self.simulate_objects(timestep)

    def simulate(self):
        for timestep in range(1, self.max_steps):
            self.step_once(timestep)

    def update_weights(self):
        # Hidden to motor neuron weights
        for i in range(self.n_springs):
            for j in range(N_HIDDEN_NEURONS):
                self.weights_hm[i, j] -= LEARNING_RATE * self.weights_hm.grad[i, j]

        # Sensor to hidden neuron weights
        for i in range(N_HIDDEN_NEURONS):
            for j in range(self.n_points):
                self.weights_sh[i, j] -= LEARNING_RATE * self.weights_sh.grad[i, j]

        # Sensor to hidden neurons
        for h in range(N_HIDDEN_NEURONS):
            # Central pattern generators
            for s in ti.static(range(N_SIN_WAVES)):
                self.weights_sh[h, s] -= LEARNING_RATE * self.weights_sh.grad[h, s]
            # 4 sensors for each object
            for j in ti.static(range(self.n_points)):
                self.weights_sh[h, N_SIN_WAVES + 4*j] -= LEARNING_RATE * self.weights_sh[h, N_SIN_WAVES + 4*j]
                self.weights_sh[h, N_SIN_WAVES + 4*j + 1] -= LEARNING_RATE * self.weights_sh[h, N_SIN_WAVES + 4*j + 1]
                self.weights_sh[h, N_SIN_WAVES + 4*j + 2] -= LEARNING_RATE * self.weights_sh[h, N_SIN_WAVES + 4*j + 2]
                self.weights_sh[h, N_SIN_WAVES + 4*j + 3] -= LEARNING_RATE * self.weights_sh[h, N_SIN_WAVES + 4*j + 3]


            self.weights_sh[h, self.n_points*4 + N_SIN_WAVES] -= LEARNING_RATE * self.weights_sh[h, self.n_points*4 + N_SIN_WAVES]
            self.weights_sh[h, self.n_points*4 + N_SIN_WAVES + 1] -= LEARNING_RATE * self.weights_sh[h, self.n_points*4 + N_SIN_WAVES + 1]

        # Hidden layer biases
        for i in range(N_HIDDEN_NEURONS):
            self.hidden_neuron_bias[i] -= LEARNING_RATE * self.hidden_neuron_bias.grad[i]
        # Motor neuron biases
        for i in range(self.n_springs):
            self.motor_neuron_bias[i] -= LEARNING_RATE * self.motor_neuron_bias.grad[i]

        

    def optimize_brain(self, n_iters=10):
        for iteration in range(n_iters): 
            self.run_simulation()
            self.update_weights()
            print(f'Iteration {iteration} loss: {self.loss}')

        return self.loss

    # Draw
    def draw(self, frame_offset):
        for timestep in range(self.max_steps):
            gui = ti.GUI('robot', (512,512), show_gui=False, background_color=0xFFFFFF)
            gui.line(begin = (0, ground_height),
                    end = (1, ground_height),
                    color=0x0,
                    radius=3)

            for object_idx in range(self.n_points):
                x,y = self.positions[timestep, object_idx]
                gui.circle((x,y), color=0x0, radius=5)

            for spring_idx in range(self.n_springs):
                object_a_idx = self.spring_anchor_a[spring_idx]
                object_b_idx = self.spring_anchor_b[spring_idx]
                position_a = self.positions[timestep, object_a_idx]
                position_b = self.positions[timestep, object_b_idx]
                if self.springs[spring_idx][3] == True:
                    gui.line(position_a, position_b, color=0x1, radius=3)
                else:
                    gui.line(position_a, position_b, color=0x1)



            gui.show('test' + str(frame_offset + timestep) + '.png')

    def make_and_watch_movie(self, file_name):
        os.system(f'rm {file_name}')
        os.system(f'ffmpeg -i test%d.png {file_name}')
        mp4 = open(file_name, 'rb').read()
        data_url = 'data:video/mp4;base64,' + b64encode(mp4).decode()
        return HTML('<video width=512 controls> <source src="%s" type="video/mp4"></video>' % data_url)

    # --------------------------------

    def run_simulation(self):
        # for i in range(ITERS):
        with ti.ad.Tape(self.loss):
            self.simulate()

            self.loss[None] = 0
            self.compute_loss()

        return self.loss

    # --------------------------------