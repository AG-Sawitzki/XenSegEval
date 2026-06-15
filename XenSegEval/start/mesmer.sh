#!/bin/bash
#
#
. get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
#
. ~/.bashrc
#
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"./.pixi/envs/deepcell/lib"
#
pixi run -e deepcell python -m XenSegEval.segmenting.seg_mesmer -c $CONFIG_FILE