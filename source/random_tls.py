from generic_tls import *
import random

class RandomTLS(GenericTLS):

    def __init__(self, tls_id):
        super().__init__(tls_id)

    def run_simulation(self, sim_file, num_steps, output_file):

        out = open(output_file, 'w')

        self.start_simulation(sim_file)

        red_waiting_time = 0
        yellow_waiting_time = 0
        green_waiting_time = 0

        total_num_vehicles = 0

        num_tls_phases = len(self.get_tls_all_phases())
        random_phase_index = 0  # start from phase 0

        for step in range(num_steps):
            self.do_one_simulation_step()

            # choose random phase every 10 steps. closer to reality. traffic lights phases don't change this fast
            # if we replace to choose randomly every 1 step, we get much better results.
            # probabily an issue with the simulation, try more realistic simulation ?
            if (step%10 == 0):
                # assuming from every state we have 2 possibel actions :
                # 1- stay in current phase
                # 2- go to next phase
                random_phase_index = random.choice([random_phase_index, (random_phase_index+1)%num_tls_phases])
                self.set_tls_phase(random_phase_index)

            total_num_vehicles += len(traci.vehicle.getIDList())

            for vehicle_id, distance_to_tls, tls_color in self.get_waiting_vehicles_on_tls():
                red_waiting_time += 1 if tls_color == 'r' else 0
                yellow_waiting_time += 1 if tls_color == 'y' else 0
                green_waiting_time += 1 if (tls_color == 'g' or tls_color == 'G') else 0

        out.write(f"tls_id = {self.tls_id}\n")
        out.write(f"red_waiting_time = {red_waiting_time}\n")
        out.write(f"yellow_waiting_time = {yellow_waiting_time}\n")
        out.write(f"green_waiting_time = {green_waiting_time}\n") # should be 0
        out.write(f"average num alive vehicles = {total_num_vehicles/num_steps}\n")

        self.end_simulation()
        out.close()

        total_waiting_time = red_waiting_time + yellow_waiting_time + green_waiting_time
        return total_waiting_time

    