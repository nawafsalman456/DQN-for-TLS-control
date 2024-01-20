import os
import sys
sys.path.append(f"{os.environ.get('PROJECT_ROOT')}\source")
from random_tls import *

if __name__ == "__main__":
    tls_id = 'my_traffic_light'
    random_tls = RandomTLS(tls_id)

    sim_file = f"{os.environ.get('PROJECT_ROOT')}\\verif\sim\\try\\try.sumocfg"
    num_steps = 1000

    target_dir = f"{os.environ.get('PROJECT_ROOT')}\\target"
    os.makedirs(target_dir, exist_ok=True)
    output_file = f"{target_dir}\\random_tls.out"

    random_tls.run_simulation(sim_file, num_steps, output_file)