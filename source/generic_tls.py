import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary

class GenericTLS:

    def __init__(self, tls_id):
        self.tls_id = tls_id

    def start_simulation(self, scenario_file):
        sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, '-c', scenario_file])

    def end_simulation(self):
        traci.close()

    def do_one_simulation_step(self):
        traci.simulationStep()

    # return list of vehicles waiting for this TLS, each entry:
    # (vehicle_id, distance_to_tls, tls_color)
    # tls_color : r, y, g, G
    def get_waiting_vehicles_on_tls(self):
        waiting_vehicles = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles_on_lane:
                if traci.vehicle.getSpeed(vehicle_id) == 0:
                    next_tls_tuple = traci.vehicle.getNextTLS(vehicle_id)
                    next_tls_id, tls_index, tls_distance, tls_color = next_tls_tuple[0]
                    assert(self.tls_id == next_tls_id)
                    waiting_vehicles.append((vehicle_id, tls_distance, tls_color))
        return waiting_vehicles

    def get_tls_all_phases(self):
        all_program_logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
        assert(all_program_logics != () and all_program_logics != None)
        return all_program_logics[0].getPhases()

    def set_tls_phase(self, phase_index):
        traci.trafficlight.setPhase(self.tls_id, phase_index)