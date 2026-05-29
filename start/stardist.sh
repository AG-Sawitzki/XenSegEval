#!/bin/bash
#
echo "$PWD"
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")

. ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mm_path/envs/stardist/lib/

micromamba activate stardist

python ./segmenting/seg_stardist.py -c $CONFIG_FILE