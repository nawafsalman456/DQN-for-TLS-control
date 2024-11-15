import random
from itertools import cycle
from collections import deque
import numpy as np

from generic_tls import *
import traffic_network
import matplotlib.pyplot as plt


root = os.environ.get('PROJECT_ROOT')

# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

sim_dir = None
if args.sim == "high_pressure":
    sim_dir = f"{root}/sim/high_pressure/"
elif args.sim == "low_pressure":
    sim_dir = f"{root}/sim/low_pressure/"
else:
    raise Exception("need to specify simulation type. run script with --sim <SIM_TYPE>. SIM_TYPE is either high_pressure or low_pressure")

# set constant seed
random.seed(args.seed)

sim_file = f"{sim_dir}/sumo/single_tls_4_way.sumocfg"
state = env.reset(is_gui=args.gui, collect_data=args.plot_space_time, sim_file=sim_file)

env.MIN_TIME_IN_PHASE = 10
env.MAX_TIME_IN_PHASE = 50

ALL_RED = ""
for _ in range(env.tls.get_num_lanes()):
    ALL_RED += "r"

rewards = []
non_weighted_rewards = []

num_runs = 1
for r in range(num_runs):
    random.seed(args.seed + r*1234321)
    total_reward = 0
    total_non_weighted_reward = 0
    curr_phase = 0
    time_in_yellow = 0
    time_in_red = 0
    env.reset(sim_file=sim_file, collect_data=args.plot_space_time)
    while True:
        
        curr_colors = env.tls.get_curr_colors()
        curr_phase = env.tls.get_curr_phase()
        next_phase = env.tls.max_pressure()
        
        is_yellow_phase = ("y" in curr_colors)
        is_red_phase = (curr_colors == ALL_RED)
        is_green_phase = not(is_yellow_phase) and not(is_red_phase)

        if next_phase != curr_phase and env.tls.get_curr_phase_spent_time() < env.MIN_TIME_IN_PHASE and is_green_phase:
            next_phase = curr_phase
            
        # Force change phase if max time in the current phase is exceeded
        if next_phase == curr_phase and env.tls.get_curr_phase_spent_time() > env.MAX_TIME_IN_PHASE:
            env.tls.green_phases_mask[curr_phase] = 0
            if not (1 in env.tls.green_phases_mask):
                env.tls.green_phases_mask = env.tls.get_tls_green_phases_mask()
            next_phase = curr_phase + 1

        if next_phase != curr_phase and is_green_phase:
            next_phase = curr_phase + 1
            time_in_yellow = 1
            
        if is_yellow_phase:
            if time_in_yellow < 2:
                next_phase = curr_phase
                time_in_yellow += 1
            else:
                next_phase = curr_phase + 1
                time_in_yellow = 0
                
        env.tls.set_max_pressure_tls_phase(next_phase)
        # do the action
        env.tls.do_one_simulation_step()

        if env.collect_data:
            env.tls.aggregate_data()

        env.curr_step += 1
        num_cars, num_buses = env.tls.get_total_num_cars_and_buses()
        weighted_reward = -(num_cars * env.CAR_WEIGHT + num_buses * env.BUS_WEIGHT)
        non_weighted_reward = -(num_cars + num_buses)
        total_reward += weighted_reward
        total_non_weighted_reward += non_weighted_reward
        is_terminated =  (env.curr_step >= env.SIM_STEPS)
        if is_terminated:
            print(f"\n\ntotal_reward = {total_reward}\n")
            print(f"total_non_weighted_reward = {total_non_weighted_reward}\n")
            rewards.append(total_reward)
            non_weighted_rewards.append(total_non_weighted_reward)
            env.terminate()
            break
    
if args.plot_space_time:
    env.plot_space_time("Max Pressure")

print("rewards = ", rewards)
print("non_weighted_rewards = ", non_weighted_rewards)

rewards_mean = np.mean(rewards, axis=0)
rewards_std = np.std(rewards, axis=0)

non_weighted_rewards_mean = np.mean(non_weighted_rewards, axis=0)
non_weighted_rewards_std = np.std(non_weighted_rewards, axis=0)

print("rewards_mean = ", rewards_mean)
print("rewards_std = ", rewards_std)

print("non_weighted_rewards_mean = ", non_weighted_rewards_mean)
print("non_weighted_rewards_std = ", non_weighted_rewards_std)

res_str = "#auto generated from max_pressure.py\n"
res_str += "max_pressure_results = {\n"
res_str += f"    \"rewards_mean\" : {rewards_mean},\n"
res_str += f"    \"rewards_std\" : {rewards_std},\n"
res_str += f"    \"non_weighted_rewards_mean\" : {non_weighted_rewards_mean},\n"
res_str += f"    \"non_weighted_rewards_std\" : {non_weighted_rewards_std}\n"
res_str += "}"

RESULTS_FILE_PATH = f"{sim_dir}/max_pressure_results/results.py"
with open(RESULTS_FILE_PATH, "w") as f:
    f.write(res_str)
