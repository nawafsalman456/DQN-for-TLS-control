import random
from itertools import cycle
from collections import deque

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

green_phases = env.tls.get_tls_green_phases()
ALL_RED = ""
for _ in range(env.tls.get_num_lanes()):
    ALL_RED += "r"

rewards = []

num_episodes = 1
for t in range(num_episodes):
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

        if next_phase != curr_phase and env.tls.get_curr_phase_spent_time() < env.MIN_TIME_IN_PHASE and not(is_yellow_phase) and not(is_red_phase):
            next_phase = curr_phase
            
        # Force change phase if max time in the current phase is exceeded
        if next_phase == curr_phase and env.tls.get_curr_phase_spent_time() > env.MAX_TIME_IN_PHASE:
            env.tls.green_phases_mask[curr_phase] = 0
            next_phase = env.tls.max_pressure()

        # print("curr_phase = ", curr_phase)
        # print("next_phase = ", next_phase)
        # print("time_in_yellow = ", time_in_yellow)
        # Implement yellow phase logic
        if not(is_red_phase) and ((next_phase != curr_phase and time_in_yellow == 0 and not(is_yellow_phase)) or (time_in_yellow == 1)):
            if is_yellow_phase and time_in_yellow == 1:
                next_phase = curr_phase
            else:
                next_phase = curr_phase + 1
            time_in_yellow += 1

        # Implement red phase logic
        if (is_yellow_phase and time_in_yellow == 0 and time_in_red == 0):
            time_in_red += 1
            next_phase = curr_phase + 1
            
        if (time_in_yellow == 2):
            time_in_yellow = 0

        if (time_in_red == 1):
            time_in_red = 0
            
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
            print(f"\ntotal_reward = {total_reward}\n")
            print(f"total_non_weighted_reward = {total_non_weighted_reward}\n")
            rewards.append(total_reward)
            env.terminate()
            plt.plot(rewards)
            plt.title('Rewards')
            plt.xlabel('episode')
            plt.ylabel('Total Reward')
            # plt.show()
            plt.savefig(f'imgs/rewards_max_pressure.png')
            break
    

print(rewards)
plt.plot(rewards)
plt.title('Rewards')
plt.xlabel('episode')
plt.ylabel('Total Reward')
# plt.show()
plt.savefig(f'imgs/rewards_max_pressure.png')

print(f"\ntotal_reward = {total_reward}\n")
env.plot_space_time(0)
