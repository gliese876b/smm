#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rlpath=$(dirname "$SCRIPT_DIR")

python3 -m venv $rlpath/rl_env

source $rlpath/rl_env/bin/activate

pip install numpy
pip install gym
pip install pympler
pip install sklearn
pip install matplotlib
pip install progressbar

deactivate
