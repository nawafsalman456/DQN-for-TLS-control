source set_env.sh

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sumo_env

python3 source/max_pressure.py --sim high_pressure

