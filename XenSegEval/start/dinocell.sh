#!/bin/bash
#
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
n_roi=$(get_toml_values "$CONFIG_FILE" "preprocessing" "n_roi")
#
#
. ~/.bashrc
pixi shell -e dinocell
#
python segmenting/dinocell.py
