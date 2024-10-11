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

# set constant seed
random.seed(args.seed)

# sim_file = f"{root}/verif/sim/try/try.sumocfg"
sim_file = f"{root}/verif/sim/single_tls_4_way/single_tls_4_way.sumocfg"
state = env.reset(is_gui=args.gui, collect_data=False, sim_file=sim_file)


env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 35

ALL_RED = ""
for _ in range(env.tls.get_num_lanes()):
    ALL_RED += "r"

rewards = []
non_weighted_rewards = []

num_runs = 10
for r in range(num_runs):
    random.seed(args.seed + r*1234321)
    total_reward = 0
    total_non_weighted_reward = 0
    curr_phase = 0
    time_in_yellow = 0
    time_in_red = 0
    env.reset(sim_file=sim_file)
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
            # next_phase = env.tls.max_pressure()

        # print("curr_phase = ", curr_phase)
        # print("next_phase = ", next_phase)
        # print("time_in_yellow = ", time_in_yellow)
        # Implement yellow phase logic
        # if not(is_red_phase) and ((next_phase != curr_phase and time_in_yellow == 0 and not(is_yellow_phase)) or (time_in_yellow == 1)):
        #     if is_yellow_phase and time_in_yellow == 1:
        #         next_phase = curr_phase
        #     else:
        #         next_phase = curr_phase + 1
        #     time_in_yellow += 1
        
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
                
        # if is_red_phase and time_in_red >= 1:
        #     time_in_red += 1
        #     next_phase = curr_phase + 1
            
            
        # Implement red phase logic
        # if (is_yellow_phase and time_in_yellow == 0 and time_in_red == 0):
            
        # # if (time_in_yellow == 2):

        # if (time_in_red == 1):
        #     time_in_red = 0
            
        # print("next_phase = ", next_phase)
        env.tls.set_max_pressure_tls_phase(next_phase)

        # do the action
        env.tls.do_one_simulation_step()

        if env.collect_data:
            env.tls.aggregate_data()

        env.curr_step += 1
        weighted_reward = -env.tls.get_weighted_num_vehicles()
        non_weighted_reward = -traci.vehicle.getIDCount()
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

RESULTS_FILE_PATH = f"{root}/max_pressure_results/results.py"
with open(RESULTS_FILE_PATH, "w") as f:
    f.write(res_str)
