import os
import time
import pickle
import math
import taichi as ti
import numpy as np
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt

from robot import SpringRobot

# Initialization
ti.init(arch=ti.metal, default_fp = ti.f32)

ground_height = 0.18
dt = 0.018
gravity = -9.8
# gravity = -19.6
sim_damping = 0.9
# spring_stiffness = 30.0
spring_stiffness = 4000.0
ITERS = 10
# LEARNING_RATE = 0.01
# LEARNING_RATE = 0.5
LEARNING_RATE = 2.0
# LEARNING_RATE = 2.0
N_HIDDEN_NEURONS = 16
N_SIN_WAVES = 2
GOAL_POSITION = [0.9,0.2]


@ti.data_oriented
class Simulation:
    def __init__(self, robot):
        self.robot = robot
        self.robot_starting_points = robot.get_sim_body_points(x_start=0.2, y_start=0.2, body_size=0.2)
        self.springs = robot.generate_body_springs(self.robot_starting_points)
        self.n_points = len(self.robot_starting_points)
        self.n_springs = len(self.springs)
        self.max_steps = 200
        self.losses = []

        self.initialize_fields()
        self.initialize_body()
        if robot.brain.hidden_neuron_bias is not None:
            self.initialize_from_brain()
        else:
            self.initialize_random_brain()

    def n_sensors(self):
        # n_sin_waves simulates central pattern generators
        return N_SIN_WAVES + 8

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
        self.net_force_on_objects = vec()
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
        ti.root.dense(ti.i, self.max_steps).dense(ti.j, self.n_points).place(self.net_force_on_objects)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_actuation)
        ti.root.dense(ti.i, self.n_springs).place(self.spring_frequency)
        ti.root.dense(ti.i, self.max_steps).place(self.center)
        ti.root.place(self.goal)

        ti.root.dense(ti.ijk, (self.max_steps, self.n_springs, N_HIDDEN_NEURONS)).place(self.hidden_neuron_values)
        ti.root.dense(ti.ij, (self.max_steps, self.n_springs)).place(self.motor_neuron_values)
        ti.root.dense(ti.ij, (self.n_springs, N_HIDDEN_NEURONS)).place(self.hidden_neuron_bias)
        ti.root.dense(ti.i, self.n_springs).place(self.motor_neuron_bias)

        ti.root.dense(ti.ijk, (self.n_springs, N_HIDDEN_NEURONS, self.n_sensors())).place(self.weights_sh)
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
                self.net_force_on_objects[i,j][0] = 0.0
                self.net_force_on_objects[i,j][1] = 0.0

        for i in range(self.max_steps):
            for j in range(self.n_springs):
                for k in range(N_HIDDEN_NEURONS):
                    self.hidden_neuron_values[i,j,k] = 0.0

        for i in range(self.max_steps):
            for j in range(self.n_springs):
                self.motor_neuron_values[i,j] = 0.0

    def initialize_random_brain(self):
        print('INITIALIZING BRAIN')
        for sp in range(self.n_springs):
            for h in range(N_HIDDEN_NEURONS):
                self.hidden_neuron_bias[sp, h] = (np.random.randn() * 2 - 1) * 0.02
                for s in range(self.n_sensors()):
                    self.weights_sh[sp, h, s] = (np.random.randn() * 2 - 1) * 0.02
            self.motor_neuron_bias[s] = (np.random.randn() * 2 - 1) * 0.02
            for h in range(N_HIDDEN_NEURONS):
                self.weights_hm[sp, h] = (np.random.randn() * 2 - 1) * 0.02

    def initialize_from_brain(self):
        for sp in range(self.n_springs):
            for h in range(N_HIDDEN_NEURONS):
                self.hidden_neuron_bias[sp, h] = self.robot.brain.hidden_neuron_bias[sp, h]
                for s in range(self.n_sensors()):
                    self.weights_sh[sp, h, s] = self.robot.brain.weights_sh[sp, h, s]

            self.motor_neuron_bias[sp] = self.robot.brain.motor_neuron_bias[sp]
            for h in range(N_HIDDEN_NEURONS):
                self.weights_hm[sp, h] = self.robot.brain.weights_hm[sp, h]


    @ti.kernel
    def compute_loss(self):
        # self.loss[None] -= self.positions[self.max_steps-1, 1][0]
        # mean x value
        # self.loss[None] -= (self.positions[self.max_steps-1, 0][0] + \
        #                     self.positions[self.max_steps-1, -1][0]) / 2
        #                     self.positions[self.max_steps-1, 2][0] + \
        #                     self.positions[self.max_steps-1, 3][0]) / 4
        
        for object_idx in range(self.n_points):
            self.loss[None] -= self.positions[self.max_steps-1, object_idx][0] / self.n_points
        # self.loss[None] -= self.center[self.max_steps-1][0]
        # self.loss[None] -= np.mean(self.motor_neuron_values[self.max_steps-1, :][0])

    @ti.func
    def simulate_neural_network_sh(self, timestep: ti.i32, spring_idx: ti.i32):
        object_a_idx = self.spring_anchor_a[spring_idx]
        object_b_idx = self.spring_anchor_b[spring_idx]
        position_a = self.positions[timestep-1, object_a_idx]
        position_b = self.positions[timestep-1, object_b_idx]
        spring_center = (position_a + position_b) / 2 # Center of the spring
        # dist = self.positions[timestep-1, object_a_idx] - self.positions[timestep-1, object_b_idx]
        # length = dist.norm()

        # NN inputs
        offset_a = spring_center - position_a # 2 x,y values
        offset_b = spring_center - position_b # 2 x,y values
        accel_a = (self.velocities[timestep-1, object_a_idx] - self.velocities[timestep-2, object_a_idx]) # 2 x,y values
        accel_b = (self.velocities[timestep-1, object_b_idx] - self.velocities[timestep-2, object_b_idx]) # 2 x,y values
        # touch_a = 1.0 if position_a[1] <= ground_height else 0.0
        # touch_b = 1.0 if position_b[1] <= ground_height else 0.0

        for h in ti.static(range(N_HIDDEN_NEURONS)):
            activation = 0.0

            # Central pattern generators
            for s in ti.static(range(N_SIN_WAVES)):
                activation += self.weights_sh[spring_idx, h, s] * ti.sin(50*timestep * dt + \
                                                        2*math.pi / N_SIN_WAVES * s)
            # Relative x,y coordinates for a and b objects
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES] * offset_a[0]     # relative x coordinate
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 1] * offset_a[1] # relative y coordinate
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 2] * offset_b[0]     # relative x coordinate
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 3] * offset_b[1] # relative y coordinate
            # Acceleration sensors:
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 4] * accel_a[0]
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 5] * accel_a[1]
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 6] * accel_b[0]
            activation += 0.1 * self.weights_sh[spring_idx, h, N_SIN_WAVES + 7] * accel_b[1]
            # Touch sensors
            # activation += 0.5*self.weights_sh[h, N_SIN_WAVES + 8] * touch_a
            # activation += 0.5*self.weights_sh[h, N_SIN_WAVES + 9] * touch_b

            activation = ti.tanh(activation + self.hidden_neuron_bias[spring_idx, h])
            self.hidden_neuron_values[timestep, spring_idx, h] = activation

    @ti.func
    def simulate_neural_network_hm(self, timestep : ti.i32, spring_idx: ti.i32):
        # for spring_idx in range(self.n_springs):
        activation = 0.0
        for h in ti.static(range(N_HIDDEN_NEURONS)):
            activation += self.weights_hm[spring_idx, h] * self.hidden_neuron_values[timestep, spring_idx, h]

        activation = ti.tanh(activation + self.motor_neuron_bias[spring_idx])
        self.motor_neuron_values[timestep, spring_idx] = activation

    @ti.kernel
    def simulate_springs(self, timestep: ti.i32):

        for object_idx in range(self.n_points):
            self.net_force_on_objects[timestep, object_idx].fill(0.0)
            self.net_force_on_objects[timestep, object_idx].fill(0.0)
        
        for spring_idx in range(self.n_springs):
            object_a_idx = self.spring_anchor_a[spring_idx]
            object_b_idx = self.spring_anchor_b[spring_idx]
            position_a = self.positions[timestep-1, object_a_idx]
            position_b = self.positions[timestep-1, object_b_idx]
            spring_center = (position_a + position_b) / 2 # Center of the spring
            dist = self.positions[timestep-1, object_a_idx] - self.positions[timestep-1, object_b_idx]
            length = dist.norm()

            self.simulate_neural_network_sh(timestep, spring_idx)
            self.simulate_neural_network_hm(timestep, spring_idx)
            
            if position_a[1] < ground_height:
                self.positions[timestep-1, object_a_idx][1] = ground_height
                # Add normal force 
                # self.net_force_on_objects[timestep, object_a_idx] += ti.Vector([0.0, -gravity])
            if position_b[1] < ground_height:
                self.positions[timestep-1, object_b_idx][1] = ground_height
                # self.net_force_on_objects[timestep, object_b_idx] += ti.Vector([0.0, -gravity])

            # Actuation: change the resting state of the spring to oscillate (open-loop)
            spring_resting_length = self.spring_resting_lengths[spring_idx] + \
                                    self.spring_actuation[spring_idx] * \
                                    self.motor_neuron_values[timestep, spring_idx] * 0.15 # 0.3 is the monolithic
                                    # 0.05*ti.sin(spring_frequency[spring_idx] * timestep*1.0)

            # Absolute value
            # spring_unhappiness = ti.abs(2*(length - spring_resting_length))
            spring_unhappiness = 8*(length - spring_resting_length) # 8 is good, with 0.3 actuation (box robot, 0.3 freq)
            self.spring_restoring_forces[timestep, spring_idx] = (dt * spring_stiffness * spring_unhappiness / length) * dist

            # Preservation of forces...
            spring_restoring_on_a = -self.spring_restoring_forces[timestep, spring_idx]
            spring_restoring_on_b = self.spring_restoring_forces[timestep, spring_idx]

            self.net_force_on_objects[timestep, object_a_idx] += spring_restoring_on_a
            self.net_force_on_objects[timestep, object_b_idx] += spring_restoring_on_b

            # old_position_a = self.positions[timestep-1, object_a_idx]
            # old_position_b = self.positions[timestep-1, object_b_idx]
            # old_velocity_a = self.velocities[timestep-1, object_a_idx]
            # old_velocity_b = self.velocities[timestep-1, object_b_idx]
            # Apply restitution force in y direction proportional to depth of penetration
            # if old_position_a[1] < ground_height:
            #     # Transfer spring restoring force to other object
            #     self.net_force_on_objects[timestep, object_b_idx] += self.spring_restoring_forces[timestep, spring_idx]
            #     # Zero out spring restoring force on object in ground
            #     self.net_force_on_objects[timestep, object_a_idx] *= ti.Vector([0.3,0.0])
            #     # Add restitution force in Y direction proportional to depth and previous velocity



            # if old_position_b[1] < ground_height:
            #     self.net_force_on_objects[timestep, object_a_idx] += -self.spring_restoring_forces[timestep, spring_idx]
            #     self.net_force_on_objects[timestep, object_b_idx] *= ti.Vector([0.3,0.0])

            #     if self.net_force_on_objects[timestep, object_a_idx][1] < 0.0: 
            #         self.net_force_on_objects[timestep, object_b_idx][1] += 
                # penetration_depth = ground_height - old_position_b[1]
                # self.net_force_on_objects[timestep, object_b_idx][1] += penetration_depth * 1000
                # if old_velocity_b[1] < 0:
                #     self.net_force_on_objects[timestep, object_b_idx][1] += -old_velocity_b[1] * 10

                # self.net_force_on_objects[timestep, object_a_idx][0] += -0.1*self.net_force_on_objects[timestep, object_a_idx][0]
                # self.net_force_on_objects[timestep, object_a_idx][1] += -0.9*self.net_force_on_objects[timestep, object_a_idx][1]



            # if old_position_a[1] < ground_height: # and self.net_force_on_objects[timestep, object_a_idx][1] < 0.0: # Transfer y direction force to other object
            #     # Force on object connected to object in ground
            #     self.net_force_on_objects[timestep, object_b_idx][0] += -0.1*self.net_force_on_objects[timestep, object_a_idx][0]
            #     self.net_force_on_objects[timestep, object_b_idx][1] += -0.9*self.net_force_on_objects[timestep, object_a_idx][0]
            #     # Force on object in ground
            #     self.net_force_on_objects[timestep, object_a_idx][0] += -0.1*self.net_force_on_objects[timestep, object_a_idx][0]
            #     self.net_force_on_objects[timestep, object_a_idx][1] += -0.9*self.net_force_on_objects[timestep, object_a_idx][1]
            #     # self.net_force_on_objects[timestep, object_a_idx][1] += -0.8*self.net_force_on_objects[timestep, object_a_idx][1]
            # if old_position_b[1] < ground_height: # and self.net_force_on_objects[timestep, object_b_idx][1] < 0.0: # Transfer y direction force to other object
            #     self.net_force_on_objects[timestep, object_a_idx][0] += -0.1*self.net_force_on_objects[timestep, object_b_idx][0]
            #     self.net_force_on_objects[timestep, object_a_idx][1] += -0.9*self.net_force_on_objects[timestep, object_b_idx][1]

            #     self.net_force_on_objects[timestep, object_b_idx][0] += -0.1*self.net_force_on_objects[timestep, object_b_idx][0]
            #     self.net_force_on_objects[timestep, object_b_idx][1] += -0.9*self.net_force_on_objects[timestep, object_b_idx][1]

    @ti.kernel
    def simulate_objects(self, timestep: ti.i32):
        for object_idx in range(self.n_points):
            old_position = self.positions[timestep-1, object_idx]

                # old_velocity = ti.Vector([-old_velocity[0],-old_velocity[1]])

            # Update velocity...
            old_velocity = (1-sim_damping) * self.velocities[timestep-1, object_idx] \
            + dt*(self.net_force_on_objects[timestep, object_idx]) # - self.net_force_on_objects[timestep-1, object_idx])

            if old_position[1] > ground_height:
                old_velocity += dt*gravity * ti.Vector([0,1])

            if old_position[1] <= ground_height and old_velocity[1] < 0:
                # old_position = ti.Vector([old_position[0], ground_height])

                # Friction x direction, restitution y direction
                # old_velocity = ti.Vector([0.5*old_velocity[0],-0.8*old_velocity[1]])

                # ZERO Friction x direction, restitution y direction
                # old_velocity = ti.Vector([0.0,-0.8*old_velocity[1]])

            #     # ZERO friction, perfect restitution
                # old_velocity = ti.Vector([0.0,-old_velocity[1]])
                  # Friction x direction, PERFECT restitution y direction
                # old_velocity = ti.Vector([0.1*old_velocity[0],-0.9*old_velocity[1]])

                old_velocity = ti.Vector([0.0,0.0])
                # old_velocity = ti.Vector([0.1*old_velocity[0],0.0])

            new_position = old_position + dt * old_velocity
            new_velocity = old_velocity

            self.positions[timestep, object_idx] = new_position
            self.velocities[timestep, object_idx] = new_velocity

    @ti.kernel
    def compute_center(self, timestep : ti.i32):
        for object_idx in range(self.n_points):
            if object_idx == 0:
                self.center[timestep].fill(0.0)
            self.center[timestep] += self.positions[timestep-1, object_idx] / self.n_points

    # Simulate
    def step_once(self, timestep : ti.i32):
        self.compute_center(timestep)
        # if timestep == 1:
        #     print(self.center[1])
        #     # print(self.positions[timestep, 0])
        #     print(self.positions[0, 0])
        #     print(self.velocities[0, 0])
        

        self.simulate_springs(timestep)
        # print(self.motor_neuron_values[timestep, 0])

        self.simulate_objects(timestep)

    def simulate(self):
        for timestep in range(1, self.max_steps):
            self.step_once(timestep)

    def update_weights(self):
        for sp in ti.static(range(self.n_springs)):
            # Hidden to motor neuron weights
            for j in range(N_HIDDEN_NEURONS):
                self.weights_hm[sp, j] -= LEARNING_RATE * self.weights_hm.grad[sp, j]

            # Sensor to hidden neuron weights
            for i in range(N_HIDDEN_NEURONS):
                for j in range(self.n_points):
                    self.weights_sh[sp, i, j] -= LEARNING_RATE * self.weights_sh.grad[sp, i, j]

            # Sensor to hidden neurons
            for h in range(N_HIDDEN_NEURONS):
                # Central pattern generators
                for s in ti.static(range(N_SIN_WAVES)):
                    self.weights_sh[sp, h, s] -= LEARNING_RATE * self.weights_sh.grad[sp, h, s]
                self.weights_sh[sp, h, N_SIN_WAVES] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES]
                self.weights_sh[sp, h, N_SIN_WAVES + 1] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 1]
                self.weights_sh[sp, h, N_SIN_WAVES + 2] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 2]
                self.weights_sh[sp, h, N_SIN_WAVES + 3] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 3]
                self.weights_sh[sp, h, N_SIN_WAVES + 4] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 4]
                self.weights_sh[sp, h, N_SIN_WAVES + 5] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 5]
                self.weights_sh[sp, h, N_SIN_WAVES + 6] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 6]
                self.weights_sh[sp, h, N_SIN_WAVES + 7] -= LEARNING_RATE * self.weights_sh.grad[sp, h, N_SIN_WAVES + 7]
                # self.weights_sh[h, N_SIN_WAVES + 8] -= LEARNING_RATE * self.weights_sh.grad[h, N_SIN_WAVES + 8]
                # self.weights_sh[h, N_SIN_WAVES + 9] -= LEARNING_RATE * self.weights_sh.grad[h, N_SIN_WAVES + 9]

                # self.weights_sh[h, self.n_points*4 + N_SIN_WAVES] -= LEARNING_RATE * self.weights_sh[h, self.n_points*4 + N_SIN_WAVES]
                # self.weights_sh[h, self.n_points*4 + N_SIN_WAVES + 1] -= LEARNING_RATE * self.weights_sh[h, self.n_points*4 + N_SIN_WAVES + 1]

            # Hidden layer biases
            for i in range(N_HIDDEN_NEURONS):
                self.hidden_neuron_bias[sp, i] -= LEARNING_RATE * self.hidden_neuron_bias.grad[sp, i]
            # Motor neuron biases
            # for i in range(self.n_springs):
            self.motor_neuron_bias[sp] -= LEARNING_RATE * self.motor_neuron_bias.grad[sp]

        

    def optimize_brain(self, n_iters=10):
        for iteration in range(n_iters): 
            start = time.time()
            # self.initialize_fields()
            # self.initialize_body()
            self.run_simulation()
            self.losses.append(self.loss.to_numpy())
            if np.isnan(self.loss.to_numpy()):
                print('Loss is NaN')
                break
            # print(iteration, n_iters - 1)
            if iteration < n_iters-1:
                print('Updating weights...')
                self.update_weights()
            lapsed = time.time() - start
            print(f'Iteration {iteration} loss: {self.loss} ({lapsed} seconds)')
        
        self.robot.brain.hidden_neuron_bias = self.hidden_neuron_bias.to_numpy()
        self.robot.brain.motor_neuron_bias = self.motor_neuron_bias.to_numpy()
        self.robot.brain.weights_sh = self.weights_sh.to_numpy()
        self.robot.brain.weights_hm = self.weights_hm.to_numpy()
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
                # if object_idx <= 1: # paint red
                #     gui.circle((x,y), color=0xFF0000, radius=5)
                # else:
                gui.circle((x,y), color=0xFF0000, radius=5)

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
        os.system(f'rm test*.png')
        return HTML('<video width=512 controls> <source src="%s" type="video/mp4"></video>' % data_url)

    # --------------------------------

    def save_robot(self, filename):
        with open(filename, 'wb') as pf:
            pickle.dump(self, pf)

    # --------------------------------

    def run_simulation(self):
        # for i in range(ITERS):
        with ti.ad.Tape(self.loss):
            self.simulate()

            self.loss[None] = 0
            self.compute_loss()

        # spring_idx = 0
        # plt.plot(self.spring_restoring_forces.to_numpy()[:,0,0])
        # # print('Motor neuron outputs:')
        # print(self.motor_neuron_values.to_numpy()[-5:,0])
        # print(self.motor_neuron_values.to_numpy()[-5:,1])
        # print(self.motor_neuron_values.to_numpy()[-5:,2])
        # print(self.motor_neuron_values.to_numpy()[-5:,3])
        # plt.plot(self.motor_neuron_values.to_numpy()[:,0])
        # plt.plot(self.motor_neuron_values.to_numpy()[:,1])
        # plt.plot(self.motor_neuron_values.to_numpy()[:,2])

        # print(self.weights_sh.to_numpy()[:, 0])
        # print('Hidden neuron values:')
        # print(self.hidden_neuron_values.to_numpy()[-5:, 0, 0])
        # print(self.hidden_neuron_values.to_numpy()[-5:, 1, 0])
        # print(self.hidden_neuron_values.to_numpy()[-5:, 2, 0])
        # print(self.hidden_neuron_values.to_numpy()[-5:, 3, 0])
        # plt.plot(self.hidden_neuron_values.to_numpy()[:, 0, 0])
        # plt.plot(self.hidden_neuron_values.to_numpy()[:, 1, 0])
        # plt.plot(self.hidden_neuron_values.to_numpy()[:, 2, 0])
        # print('Hidden neuron --> Motor neuron weights:')
        # print(self.weights_hm.to_numpy()[0, :5])
        # print(self.weights_hm.to_numpy()[1, :5])
        # print(self.weights_hm.to_numpy()[2, :5])

        # Plot the object x positions over time 
        # for object_idx in range(self.n_points):
        #     plt.plot(self.positions.to_numpy()[:, object_idx, 0])
        # plt.plot(self.spring_resting_lengths[spring_idx] + self.spring_actuation[spring_idx] * self.motor_neuron_values.to_numpy()[:, spring_idx])

        return self.loss

    # --------------------------------