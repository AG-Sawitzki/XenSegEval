#!/bin/bash

source get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
bashrc_path=$(get_toml_value "$CONFIG_FILE" "paths" "bashrc_path")

source $bashrc_path
micromamba activate stardist

python segmenting/seg_stardist.py