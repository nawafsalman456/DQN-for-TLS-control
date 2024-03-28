import traci
import traci.constants as tc
import os
import sys
import random
import numpy as np
from generic_tls import *
import traffic_network

# Function to calculate pressure based on queue lengths
def calculate_pressure(queue):
    pressures = []
    for q in queue:
        pressure = q * 1  # Example pressure function
        pressures.append(pressure)
    return pressures

# # Function to calculate weights based on pressure
# def calculate_weights(queue, pressure, detector):
#     weights = np.zeros((len(queue), len(queue)))
#     for i in range(len(queue)):
#         for j in range(len(queue)):
#             if i != j:
#                 weights[i, j] = detector[i][j] * max(pressure[i] - pressure[j], 0)
#     return weights

# Function to calculate the optimal phase using BP algorithm
def calculate_optimal_phase(queue, detector, coefficient):
    pressure = calculate_pressure(queue)
    weights = pressure#calculate_weights(queue, pressure, detector)
    best_phase = None
    best_weight_sum = -float('inf')
    for phase, coeff in enumerate(coefficient):
        weight_sum = np.sum(weights * coeff)
        if weight_sum > best_weight_sum:
            best_weight_sum = weight_sum
            best_phase = phase
    return best_phase


root = os.environ.get('PROJECT_ROOT')

# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

# set constant seed
random.seed(args.seed)

state = env.reset(is_gui=args.gui)

total_reward = 0

env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 20

coefficient = [0]*len(env.tls.get_curr_phase_encoding())

while True:

    # num_vehicles = env.tls.get_num_vehicles_on_each_lane()

    # # Calculate back-pressure based on num_vehicles
    # back_pressure = sum(num_vehicles)

    # action = 0  # by default, stay in current state

    # # Set action based on back-pressure
    # if back_pressure > threshold:
    #     action = 1

    queue = env.tls.get_num_vehicles_on_each_lane()
    detector = None

    curr_phase_index = env.tls.get_curr_phase()
    coefficient[curr_phase_index] = env.tls.get_curr_phase_spent_time()
    
    optimal_phase = calculate_optimal_phase(queue, detector, coefficient)
    action = 1 if optimal_phase > curr_phase_index else 0

    observation, reward, terminated = env.step(action)
    total_reward += reward

    if terminated:
        break

print(f"\ntotal_reward = {total_reward}\n")
# env.plot_space_time(0)
