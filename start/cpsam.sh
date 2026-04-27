#!/bin/bash

source /path/to/lib_ini.sh
CONFIG_FILE=$1
home=$(ini_read "$CONFIG_FILE" "PATHS" "home")
sample=$(ini_read "$CONFIG_FILE" "PATHS" "sample_name")
bashrc_path=$(ini_read "$CONFIG_FILE" "PATHS" "bashrc_path")

source $bashrc_path
micromamba activate cellpose

for q in {0..3}; do
	python -m cellpose \
	--image_path $home/$sample/processed/$section/morphology/multi_layer/quatered/q0$q.ome.tif \
	--do_3D --flow3D_smooth 2 \
	--save_tif --save_flow --save_outlines --in_folders \
	--savedir $home/$sample/results/cpsam/ \
	--use_gpu --verbose
	
done