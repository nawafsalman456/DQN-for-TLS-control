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

env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 30
MAX_DISTANCE = 100

while True:

    max_lane_pressure = -1
    max_pressure_lane = 0

    num_vehicles_on_each_lane = env.tls.get_num_vehicles_on_each_lane_with_limited_distance(MAX_DISTANCE)
    wait_time_on_each_lane = env.tls.get_waiting_time_on_each_lane()
    for lane_id, num_vehicles in enumerate(num_vehicles_on_each_lane):
        pressure = num_vehicles + 0.5 * wait_time_on_each_lane[lane_id]
        if pressure > max_lane_pressure:
            max_lane_pressure = pressure
            max_pressure_lane = lane_id

    optimal_phase_index = 0
    num_green_lanes_in_optimal_phase = 0

    all_phases = env.tls.get_tls_all_phases()
    for phase_index, phase in enumerate(all_phases):
        colors_string = phase.state
        # Get lanes with colors "g" and "G"
        lanes_with_green_color = [i for i, color in enumerate(colors_string) if color.upper() == "G"]
        if (max_pressure_lane in lanes_with_green_color) and (len(lanes_with_green_color) >= num_green_lanes_in_optimal_phase):
            optimal_phase_index = phase_index
            num_green_lanes_in_optimal_phase = len(lanes_with_green_color)

    # curr_phase = env.tls.get_curr_phase()
    # action = 1 if optimal_phase_index > curr_phase else 0
    env.tls.set_tls_phase(optimal_phase_index)
    observation, reward, terminated = env.step()#action)
    total_reward += reward

    if terminated:
        break

print(f"\ntotal_reward = {total_reward}\n")
# env.plot_space_time(0)
