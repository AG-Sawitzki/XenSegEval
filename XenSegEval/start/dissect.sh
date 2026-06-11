#!/bin/bash
#
. get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
n_roi=$(get_toml_value "$CONFIG_FILE" "preprocessing" "n_roi")
#
#
. ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/"$mm_path/dissect/lib"
#
pixi run -e dissect python XenSegEval/segmenting/seg_dissect.py -c $CONFIG_FILE