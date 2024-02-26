import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary
import generic_tls
import random
import io
import torch

root = os.environ.get('PROJECT_ROOT')

SIM_STEPS = 4000    # for now to debug the training loop. make sure it is working and replace with 3600 (1 hour)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrafficNetwork:

    def __init__(self):
        # TODO - move TLS names to test file
        self.tls = generic_tls.GenericTLS('my_traffic_light')   # TODO - for now assume 1 tls, later create a list of all TLSs in the network ??
        self.tls_curr_phase = 0  # start from phase 0

        self.vehicle_entry_times = {}
        self.curr_step = 0
        self.reward = 0
        self.state = []
        self.action_space = [0, 1]  # action space for each TLS : 0 - stay in current state. 1 - move to next state. TODO - when network contains a list of TLSs. action space will be a dict of all permutation of actions of all TLSs in network ?

    def reset(self, is_gui):
        # TODO - move below 2 to test file
        sim_file = f"{root}\\verif\sim\\try\\try.sumocfg"
        try:
            self.tls.start_simulation(sim_file, is_gui)
        except traci.exceptions.TraCIException:
            self.terminate()
            self.tls.start_simulation(sim_file, is_gui)

        self.vehicle_entry_times = {}
        self.curr_step = 0
        self.reward = 0
        self.state = self.get_curr_phase_encoding() + [self.tls.get_curr_phase_spent_time()] + self.tls.get_all_phases_waiting_vehicles() + [self.curr_step] + self.tls.get_num_vehicles_on_each_lane() + self.tls.get_min_vehicle_distance_on_each_lane()
        return self.state

    # TODO - when network contains a list of TLSs. "action" will be dict. in each entry - {"tls_id" : action}
    def step(self, action):
        MIN = 4
        MAX = 30
        time_in_curr_phase = self.tls.get_curr_phase_spent_time()
        if time_in_curr_phase > MAX:
            # if stuck in current phase for a long time, move to next phase
            action = 1
        if time_in_curr_phase < MIN:
            # stay in current phase for at least MIN seconds. to prevent fast transitions.
            action = 0

        num_tls_phases = len(self.tls.get_tls_all_phases())
        self.tls_curr_phase = (self.tls_curr_phase + action) % num_tls_phases   # if action=0 - stay in current state. if action=1 - move to next phase
        self.tls.set_tls_phase(self.tls_curr_phase)

        # do the action
        self.tls.do_one_simulation_step()

        self.curr_step += 1
        self.state = self.get_curr_phase_encoding() + [time_in_curr_phase] + self.tls.get_all_phases_waiting_vehicles() + [self.curr_step] + self.tls.get_num_vehicles_on_each_lane() + self.tls.get_min_vehicle_distance_on_each_lane()
        self.reward = -len(traci.vehicle.getIDList())
        is_terminated =  (self.curr_step >= SIM_STEPS)
        if is_terminated:
            self.terminate()
        return  (self.state, self.reward, is_terminated)

    def get_curr_phase_encoding(self):
        num_phases = len(self.tls.get_tls_all_phases())
        curr_phase_index = self.tls.get_curr_phase()
        phases_encoding_list = [0]*num_phases
        phases_encoding_list[curr_phase_index] = 1
        return phases_encoding_list

    def get_num_actions(self):
        return 2

    def sample_random_action(self):
        return random.sample(self.action_space, k=1)

    def terminate(self):
        self.tls.end_simulation()
