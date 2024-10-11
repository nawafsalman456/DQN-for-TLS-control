#!/bin/bash

# Source the environment setup script
source set_env.sh

# Function to run the Python script with a unique seed and retry up to 10 times on failure
run_simulation() {
    local seed=$1
    local use_buses_weighted_reward=$2
    local max_attempts=1
    local attempt=0
    local success=false

    while [ $attempt -lt $max_attempts ] && [ "$success" = false ]; do
        if [ "$use_buses_weighted_reward" = true ]; then
            python3 source/RL_tls.py --load_model --save_model --print_reward --plot_rewards --buses_weighted_reward --seed "$seed"
        else
            python3 source/RL_tls.py --load_model --save_model --print_reward --plot_rewards --seed "$seed"
        fi

        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            success=true
        else
            echo "Attempt $((attempt + 1)) failed for seed $seed. Retrying..."
            attempt=$((attempt + 1))
            sleep 1  # Wait a little before retrying
        fi
    done

    if [ "$success" = false ]; then
        echo "Failed to run simulation for seed $seed after $max_attempts attempts."
    fi
}

# use_buses_weighted_reward = true
run_simulation 21345 true &
# run_simulation 78754 true &
# run_simulation 12355 true &
# run_simulation 67878 true &
# run_simulation 21354 true &
# wait

# run_simulation 35467 true &
# run_simulation 13455 true &
# run_simulation 34563 true &
# run_simulation 07894 true &
# run_simulation 96984 true &
# wait

# use_buses_weighted_reward = false
run_simulation 21345 false &
# run_simulation 78754 false &
# run_simulation 12355 false &
# run_simulation 67878 false &
# run_simulation 21354 false &
# wait

# run_simulation 35467 false &
# run_simulation 13455 false &
# run_simulation 34563 false &
# run_simulation 07894 false &
# run_simulation 96984 false &
# wait
