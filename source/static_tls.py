from generic_tls import *
import torch

class StaticTLS(GenericTLS):

    def __init__(self, tls_id):
        super().__init__(tls_id)

    # assumes that TLS type is static in the given sim
    def run_simulation(self, sim_file, num_steps, output_file):

        out = open(output_file, 'w')

        self.start_simulation(sim_file)

        red_waiting_time = 0
        yellow_waiting_time = 0
        green_waiting_time = 0

        for step in range(num_steps):
            self.do_one_simulation_step()

            print("time_loss = ", self.get_time_loss())
            print("num_empty_lanes = ", self.get_num_empty_lanes())
            
            for vehicle_id, distance_to_tls, tls_color in self.get_waiting_vehicles_on_tls():
                red_waiting_time += 1 if tls_color == 'r' else 0
                yellow_waiting_time += 1 if tls_color == 'y' else 0
                green_waiting_time += 1 if (tls_color == 'g' or tls_color == 'G') else 0
        out.write(f"tls_id = {self.tls_id}\n")
        out.write(f"red_waiting_time = {red_waiting_time}\n")
        out.write(f"yellow_waiting_time = {yellow_waiting_time}\n")
        out.write(f"green_waiting_time = {green_waiting_time}\n") # should be 0

        self.end_simulation()
        out.close()

        total_waiting_time = red_waiting_time + yellow_waiting_time + green_waiting_time
        return total_waiting_time
    