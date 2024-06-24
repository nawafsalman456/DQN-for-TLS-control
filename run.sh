source set_env.sh

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sumo_env

python3 source/RL_tls.py --load_model --save_model --print_reward --plot_rewards