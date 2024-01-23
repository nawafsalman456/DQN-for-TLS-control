import os
import sys

root = os.environ.get('PROJECT_ROOT')
target = os.environ.get('PROJECT_TARGET')

sys.path.append(f"{root}\source")
from random_tls import *

if __name__ == "__main__":
    tls_id = 'my_traffic_light'
    random_tls = RandomTLS(tls_id)

    sim_file = f"{root}\\verif\sim\\try\\try.sumocfg"
    num_steps = 1000

    output_file = f"{target}\\random_tls.out"

    random_tls.run_simulation(sim_file, num_steps, output_file)