#!/bin/bash

source get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
bashrc_path=$(get_toml_value "$CONFIG_FILE" "paths" "bashrc_path")

source $bashrc_path
micromamba activate cpsam

for q in {0..3}; do
	python -m cellpose \
	--image_path $home/$sample/processed/$SECTION/morphology/multi_layer/quatered/q0$q.ome.tif \
	--do_3D --flow3D_smooth 2 \
	--save_tif --save_flow --save_outlines --in_folders \
	--savedir $home/$sample/results/cpsam/ \
	--use_gpu --verbose
	
done