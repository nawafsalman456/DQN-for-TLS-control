#!/bin/bash

export SUMO_HOME=/home/user_136/sumo

export PROJECT_ROOT=$(pwd)
echo "Set environment variable: PROJECT_ROOT=$PROJECT_ROOT"

export PROJECT_TARGET=$PROJECT_ROOT/target
echo "Set environment variable: PROJECT_TARGET=$PROJECT_TARGET"

if [ ! -d "$PROJECT_TARGET" ]; then
    echo "Create target dir: $PROJECT_TARGET"
    mkdir -p "$PROJECT_TARGET"
fi

echo "Verify SUMO_HOME is defined:"
if [ -n "$SUMO_HOME" ]; then
    echo "SUMO_HOME is defined: $SUMO_HOME"
else
    echo "ERROR - SUMO_HOME is not defined. Need to install SUMO simulator and make sure that SUMO_HOME is defined."
    exit 1
fi

echo "set_env completed successfully!"