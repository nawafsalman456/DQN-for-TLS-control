import traci
import traci.constants as tc
import os
import sys
import random
from generic_tls import *
import traffic_network

root = os.environ.get('PROJECT_ROOT')

# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

# set constant seed
random.seed(args.seed)

state = env.reset(is_gui=args.gui)

total_reward = 0

# HyperParameters
threshold = 45
env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 15

while True:

    num_vehicles = env.tls.get_num_vehicles_on_each_lane()

    # Calculate back-pressure based on num_vehicles
    back_pressure = sum(num_vehicles)

    action = 0  # by default, stay in current state

    # Set action based on back-pressure
    if back_pressure > threshold:
        action = 1

    observation, reward, terminated = env.step(action)
    total_reward += reward

    if terminated:
        break

print(f"\ntotal_reward = {total_reward}\n")
env.plot_space_time(0)
