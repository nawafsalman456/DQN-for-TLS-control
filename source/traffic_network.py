import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary
import generic_tls
import random
import io
import torch

root = os.environ.get('PROJECT_ROOT')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

class TrafficNetwork:

    def __init__(self):
        self.SIM_STEPS = 3600
        self.MIN_TIME_IN_PHASE = 5
        self.MAX_TIME_IN_PHASE = 50
        self.TIME_IN_YELLOW = 2
        self.TIME_IN_RED = 1
        self.CAR_WEIGHT = 1
        self.BUS_WEIGHT = 15
        self.tls = generic_tls.GenericTLS('my_traffic_light')
        self.tls_curr_phase = 0  # start from phase 0
        self.collect_data = False

        self.curr_step = 0
        self.reward = 0
        self.weighted_reward = 0
        self.non_weighted_reward = 0
        self.num_cars_on_each_lane = []
        self.num_buses_on_each_lane = []
        self.weighted_num_vehicles_on_each_lane = []
        self.num_vehicles_on_each_lane = []
        self.state = []
        self.vehicle_positions = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.action_space = [0, 1]  # action space for each TLS : 0 - stay in current state. 1 - move to next state.

    def reset(self, is_gui=False, collect_data=False, sim_file = f"{root}/verif/sim/single_tls_4_way/single_tls_4_way.sumocfg"):
        try:
            self.tls.start_simulation(sim_file, is_gui)
        except traci.exceptions.TraCIException:
            self.terminate()
            self.tls.start_simulation(sim_file, is_gui)

        self.vehicle_positions = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.curr_step = 0
        self.reward = 0
        self.weighted_reward = 0
        self.non_weighted_reward = 0
        self.num_lanes = len(traci.trafficlight.getControlledLanes(self.tls.tls_id))
        self.num_cars_on_each_lane = [0] * self.num_lanes
        self.num_buses_on_each_lane = [0] * self.num_lanes
        self.weighted_num_vehicles_on_each_lane = [0] * self.num_lanes
        self.num_vehicles_on_each_lane = [0] * self.num_lanes
        self.state = self.get_curr_state()
        self.collect_data = collect_data
        self.counter = 0
        return self.state

    def step(self, action = None, buses_weighted_reward=None):
        current_colors = self.tls.get_curr_colors()
        time_in_curr_phase = self.tls.get_curr_phase_spent_time()
        num_tls_phases = len(self.tls.get_tls_all_phases())

        if ("y" in current_colors): # if yellow phase, stay for 2 seconds
            MIN = self.TIME_IN_YELLOW
            MAX = self.TIME_IN_YELLOW
        elif ("G" in current_colors or "g" in current_colors):
            MIN = self.MIN_TIME_IN_PHASE
            MAX = self.MAX_TIME_IN_PHASE
        else:
            # if red phase, stay for 1 seconds
            MIN = self.TIME_IN_RED
            MAX = self.TIME_IN_RED

        if time_in_curr_phase >= MAX:
            # if stuck in current phase for a long time, move to next phase
            action = 1
        if time_in_curr_phase < MIN:
            # stay in current phase for at least MIN seconds. to prevent fast transitions.
            action = 0

        if action is not None:
            tls_next_phase = (self.tls_curr_phase + action) % num_tls_phases  # if action=0 - stay in current state. if action=1 - move to next phase
            self.tls.set_tls_phase(tls_next_phase)

        # do the action
        self.tls.do_one_simulation_step()
        self.tls_curr_phase = self.tls.get_curr_phase()

        # collect_data is false in training, to save latency.
        if self.collect_data:
            self.tls.aggregate_data()
            
        self.num_cars_on_each_lane, self.num_buses_on_each_lane = self.tls.get_num_cars_and_buses_on_each_lane()
        # self.weighted_num_vehicles_on_each_lane = self.get_weighted_num_vehicles_on_each_lane()
        # self.num_vehicles_on_each_lane = self.get_num_vehicles_on_each_lane()

        total_num_cars, total_num_buses = self.tls.get_total_num_cars_and_buses()

        self.curr_step += 1
        self.state = self.get_curr_state(buses_weighted_reward)  # build the state. state is the input of DQN
        self.weighted_reward = -(total_num_cars * self.CAR_WEIGHT + total_num_buses * self.BUS_WEIGHT)
        self.non_weighted_reward = -(total_num_cars + total_num_buses)
        if buses_weighted_reward:
            self.reward = self.weighted_reward
        else:
            self.reward = self.non_weighted_reward
        is_terminated =  (self.curr_step >= self.SIM_STEPS)
        if is_terminated:
            self.terminate()
        return  (self.state, self.reward, is_terminated)
    
    def get_weighted_reward(self):
        return self.weighted_reward
    
    def get_non_weighted_reward(self):
        return self.non_weighted_reward
    
    def get_weighted_num_vehicles_on_each_lane(self):
        weighted_num_vehicles = []
        for i in range(self.num_lanes):
            weighted_num_vehicles.append(self.num_cars_on_each_lane[i] * self.CAR_WEIGHT + self.num_buses_on_each_lane[i] * self.BUS_WEIGHT)
        return weighted_num_vehicles
    
    def get_num_vehicles_on_each_lane(self):
        num_vehicles = []
        for i in range(self.num_lanes):
            num_vehicles.append(self.num_cars_on_each_lane[i] + self.num_buses_on_each_lane[i])
        return num_vehicles
    
    def get_curr_state(self, buses_weighted_reward=False):
        state_list = self.tls.get_curr_phase_encoding() + \
                     [self.tls.get_curr_phase_spent_time()] + \
                     self.num_cars_on_each_lane + \
                     self.num_buses_on_each_lane
        # if buses_weighted_reward:
        #     state_list += self.weighted_num_vehicles_on_each_lane
        # else:
        #     state_list += self.num_vehicles_on_each_lane
        return state_list

    def get_num_actions(self):
        return 2

    def sample_random_action(self):
        return random.sample(self.action_space, k=1)

    def terminate(self):
        self.tls.end_simulation()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--sim', type=str, default=None)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--gui', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--save_model', action='store_true')
        parser.add_argument('--load_model', action='store_true')
        parser.add_argument('--print_reward', action='store_true')
        parser.add_argument('--plot_rewards', action='store_true')
        parser.add_argument('--plot_space_time', action='store_true')
        parser.add_argument('--buses_weighted_reward', action='store_true')
        parser.add_argument('--plot_mean_and_std', action='store_true')
        parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation (default: 42)")
        return parser.parse_args()

    def plot_space_time(self, pause_time = 10):
        # Data collection of the last simulation
        vehicle_positions, vehicle_velocities, distance_from_tls = self.tls.get_aggregated_data()
        # Plot space-time diagram
        plt.figure(figsize=(10, 5))
        scatter = plt.scatter(distance_from_tls, vehicle_positions, c=vehicle_velocities, cmap='viridis', marker='.')
        plt.colorbar(scatter, label='Velocity (m/s)')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Distance From TLS')
        plt.title('Space-Time Diagram')
        if pause_time == 0:
            # plt.show()
            plt.savefig(f'imgs/space_time.png')
            
        else:
            plt.pause(pause_time)    # Wait for "pause_time" second before closing the window
            plt.clf()  # Clear the current figure
            plt.close() # Close the current figure window
