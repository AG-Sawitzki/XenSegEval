#!/bin/bash
#
echo "$PWD"
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
#
source ~/.bashrc
#
micromamba activate deepcell
#
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/"$mm_path/deepcell/lib/"
#
python segmenting/seg_mesmer.py -c $CONFIG_FILE