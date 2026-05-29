#!/bin/bash
#
echo "$PWD"
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")

source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/cephfs-1/work/groups/sawitzki/users/juno12_c/micromamba/envs/deepcell/lib/

micromamba activate deepcell

python segmenting/seg_mesmer.py -c $CONFIG_FILE