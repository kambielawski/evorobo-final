import os
import math
import taichi as ti
import numpy as np
from IPython.display import HTML
from base64 import b64encode


# Initialization
ti.init(arch=ti.metal, default_fp = ti.f32)

max_steps = 100
ground_height = 0.1
dt = 0.01
gravity = -9.8
sim_damping = 0.1
spring_stiffness = 1000.0
# n_objects = 4
ITERS = 10
LEARNING_RATE = 0.1
ACTUATION_INIT_FREQ = 0.5
N_HIDDEN_NEURONS = 32
N_SIN_WAVES = 10
GOAL_POSITION = [0.9,0.2]
starting_object_positions = []

# for object_index in range(n_objects):
#   starting_object_positions.append([np.random.random(), np.random.random() * 0.9 + 0.1])

starting_object_positions.append([0.1, ground_height + 0.1])
starting_object_positions.append([0.1, ground_height + 0.2])
starting_object_positions.append([0.2, ground_height + 0.2])
starting_object_positions.append([0.2, ground_height + 0.1])

def make_spring(a,b, motor=False):
    resting_length = np.sqrt((starting_object_positions[a][0] - starting_object_positions[b][0])**2 + (starting_object_positions[a][1] - starting_object_positions[b][1])**2)
    return [a,b, resting_length, motor, ACTUATION_INIT_FREQ]

n_objects = len(starting_object_positions)

springs = []
springs.append(make_spring(0,1, motor=False)) # make spring between 0th and 1st object
springs.append(make_spring(1,2, motor=True))
springs.append(make_spring(2,3, motor=False))
springs.append(make_spring(3,0, motor=True))
springs.append(make_spring(0,2, motor=False))
springs.append(make_spring(1,3, motor=True))


n_springs = len(springs)
def n_sensors():
    # n_sin_waves simulates central pattern generators
    # each object has 4 internal sensors
    # 2 global sensors... 
    return N_SIN_WAVES + 4*n_objects + 2

spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_resting_lengths = ti.field(ti.f32)
spring_actuation = ti.field(ti.i32)
spring_frequency = ti.field(ti.f32)

vec = lambda: ti.Vector.field(2, dtype=ti.f32)
positions = vec()
velocities = vec()
spring_restoring_forces = vec()
spring_forces_on_objects = vec()
center = vec() # Center of the robot
goal = vec()

hidden_neuron_values = ti.field(ti.f32)
motor_neuron_values = ti.field(ti.f32)
hidden_neuron_bias = ti.field(ti.f32)
motor_neuron_bias = ti.field(ti.f32)
weights_sh = ti.field(ti.f32)
weights_hm = ti.field(ti.f32)

ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(positions)
ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(positions.grad)
ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(velocities)
ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(velocities.grad)
ti.root.dense(ti.i, n_springs).place(spring_anchor_a, spring_anchor_b, spring_resting_lengths)
ti.root.dense(ti.i, max_steps).dense(ti.j, n_springs).place(spring_restoring_forces)
ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(spring_forces_on_objects)
ti.root.dense(ti.i, n_springs).place(spring_actuation)
ti.root.dense(ti.i, n_springs).place(spring_frequency)
ti.root.dense(ti.i, max_steps).place(center)
ti.root.place(goal)

ti.root.dense(ti.ij, (max_steps, N_HIDDEN_NEURONS)).place(hidden_neuron_values)
ti.root.dense(ti.ij, (max_steps, n_springs)).place(motor_neuron_values)
ti.root.dense(ti.i, N_HIDDEN_NEURONS).place(hidden_neuron_bias)
ti.root.dense(ti.i, n_springs).place(motor_neuron_bias)

ti.root.dense(ti.ij, (N_HIDDEN_NEURONS, n_sensors())).place(weights_sh)
ti.root.dense(ti.ij, (n_springs, N_HIDDEN_NEURONS)).place(weights_hm)

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True) # 0-dimensional tensor (i.e. fp scalar)

@ti.kernel
def compute_loss():
    loss[None] = positions[max_steps-1, 0][1]


def initialize():
    goal[None] = GOAL_POSITION

    for h in range(N_HIDDEN_NEURONS):
        hidden_neuron_bias[h] = (np.random.randn() * 2 - 1) * 0.02
        for s in range(n_sensors()):
            weights_sh[h, s] = (np.random.randn() * 2 - 1) * 0.02

    for s in range(n_springs):
        motor_neuron_bias[s] = (np.random.randn() * 2 - 1) * 0.02
        for h in range(N_HIDDEN_NEURONS):
            weights_hm[s, h] = (np.random.randn() * 2 - 1) * 0.02

    for object_idx in range(n_objects):
       positions[0, object_idx] = starting_object_positions[object_idx]
       velocities[0, object_idx] = [0,0]

    for spring_idx in range(n_springs):
        s = springs[spring_idx]
        spring_anchor_a[spring_idx] = s[0]
        spring_anchor_b[spring_idx] = s[1]
        spring_resting_lengths[spring_idx] = s[2]
        spring_actuation[spring_idx] = s[3]
        spring_frequency[spring_idx] = s[4]


@ti.kernel
def simulate_springs(timestep: ti.i32):
    for spring_idx in range(n_springs):
        object_a_idx = spring_anchor_a[spring_idx]
        object_b_idx = spring_anchor_b[spring_idx]
        position_a = positions[timestep-1, object_a_idx]
        position_b = positions[timestep-1, object_b_idx]
        dist = position_a - position_b
        length = dist.norm()
        spring_resting_length = spring_resting_lengths[spring_idx]

        # Actuation: change the resting state of the spring to oscillate (open-loop)
        spring_resting_length = spring_resting_length + \
                                spring_actuation[spring_idx] * \
                                motor_neuron_values[timestep, spring_idx]
                                # 0.05*ti.sin(spring_frequency[spring_idx] * timestep*1.0)

        spring_unhappiness = length - spring_resting_length
        spring_restoring_forces[timestep, spring_idx] = (dt * spring_stiffness * spring_unhappiness / length) * dist

        spring_forces_on_objects[timestep, object_a_idx] += -spring_restoring_forces[timestep, spring_idx]
        spring_forces_on_objects[timestep, object_b_idx] += spring_restoring_forces[timestep, spring_idx]
    
@ti.kernel
def simulate_objects(timestep: ti.i32):
    for object_idx in range(n_objects):
        old_position = positions[timestep-1, object_idx]
        old_velocity = (1-sim_damping) * velocities[timestep-1, object_idx] \
        + dt*gravity * ti.Vector([0,1]) \
        + spring_forces_on_objects[timestep, object_idx]

        # Collision detection and resolution
        if old_position[1] <= ground_height and old_velocity[1] < 0:
            old_velocity = ti.Vector([0,0])

        new_position = old_position + dt * old_velocity
        new_velocity = old_velocity

        positions[timestep, object_idx] = new_position
        velocities[timestep, object_idx] = new_velocity

@ti.kernel
def simulate_neural_network_sh(timestep: ti.i32):
    for h in range(N_HIDDEN_NEURONS):
        activation = 0.0
        # Central pattern generators
        for s in ti.static(range(N_SIN_WAVES)):
            activation += weights_sh[h, s] * ti.sin(50*timestep * dt + \
                                                    2*math.pi / N_SIN_WAVES * s)
        # 4 sensors for each object
        for j in ti.static(range(n_objects)):
            offset = positions[timestep, j] - center[timestep]
            activation += 0.1*weights_sh[h, N_SIN_WAVES + 4*j] * offset[0]     # relative x coordinate
            activation += 0.1*weights_sh[h, N_SIN_WAVES + 4*j + 1] * offset[1] # relative y coordinate
            activation += 0.1*weights_sh[h, N_SIN_WAVES + 4*j + 2] * positions[timestep, j][0]
            activation += 0.1*weights_sh[h, N_SIN_WAVES + 4*j + 2] * positions[timestep, j][1]

        activation += weights_sh[h, n_objects*4 + N_SIN_WAVES]     * \
                                                    (goal[None][0] - center[timestep][0])
        activation += weights_sh[h, n_objects*4 + N_SIN_WAVES + 1] * \
                                                    (goal[None][1] - center[timestep][1])

        activation = ti.tanh(activation + hidden_neuron_bias[h])
        hidden_neuron_values[timestep, h] = activation    

@ti.kernel
def simulate_neural_network_hm(timestep : ti.i32):
    for spring_idx in range(n_springs):
        activation = 0.0
        for h in ti.static(range(N_HIDDEN_NEURONS)):
            activation += weights_hm[spring_idx, h] * hidden_neuron_values[timestep, h]

        activation = ti.tanh(activation) # + motor_neuron_bias[spring_idx])
        motor_neuron_values[timestep, spring_idx] = activation

@ti.kernel
def compute_center(timestep : ti.i32):
    for object_idx in range(n_objects):
        center[timestep] += positions[timestep, object_idx] / n_objects

# Simulate
def step_once(timestep : ti.i32):

    compute_center(timestep)

    simulate_neural_network_sh(timestep)
    simulate_neural_network_hm(timestep)

    simulate_springs(timestep)

    simulate_objects(timestep)




def simulate():
  for timestep in range(1, max_steps):
    step_once(timestep)

# Draw
def draw(frame_offset):
  for timestep in range(max_steps):
    gui = ti.GUI('robot', (512,512), show_gui=False, background_color=0xFFFFFF)
    gui.line(begin = (0, ground_height),
            end = (1, ground_height),
            color=0x0,
            radius=3)

    for object_idx in range(n_objects):
      x,y = positions[timestep, object_idx]
      gui.circle((x,y), color=0x0, radius=5)

    for spring_idx in range(n_springs):
      object_a_idx = spring_anchor_a[spring_idx]
      object_b_idx = spring_anchor_b[spring_idx]
      position_a = positions[timestep, object_a_idx]
      position_b = positions[timestep, object_b_idx]
      if springs[spring_idx][3] == True:
          gui.line(position_a, position_b, color=0x1, radius=3)
      else:
          gui.line(position_a, position_b, color=0x1)



    gui.show('test' + str(frame_offset + timestep) + '.png')

def make_and_watch_movie():
    os.system('rm movie.mp4')
    os.system('ffmpeg -i test%d.png movie.mp4')
    mp4 = open('movie.mp4', 'rb').read()
    data_url = 'data:video/mp4;base64,' + b64encode(mp4).decode()
    return HTML('<video width=512 controls> <source src="%s" type="video/mp4"></video>' % data_url)

# --------------------------------

def run_simulation():
    initialize()

    
    # for i in range(ITERS):
    with ti.ad.Tape(loss):
        simulate()
        compute_loss()

        # starting_object_positions[0] = starting_object_positions[0] + LEARNING_RATE * positions.grad[0,0]

        # Update the spring frequencies
        # for i in range(n_springs):
            # spring_frequency[i] = spring_frequency[i] + LEARNING_RATE * spring_frequency.grad[i]

    os.system('rm *.png')
    draw(0)

# --------------------------------

run_simulation()
make_and_watch_movie()
