#!/bin/bash
#
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
n_roi=$(get_toml_values "$CONFIG_FILE" "preprocessing" "n_roi")
#
#
. ~/.bashrc_path
micromamba activate cpsam
#
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/"$mm_path/cpsam/lib"
#
python segmenting/seg_cpsam.py -c $CONFIG_FILE -s $SECTION

# should the script make problems:
# if [ -z ${SECTION} ] 
#     then
#         for s in {0..$((n_roi-1))}; do
#             for q in {0..3}; do
# 	            python -m cellpose \
# 	            --image_path $home/$sample/processed/$s/morphology/multi_layer/quatered/q0$q.ome.tif \
# 	            --do_3D --flow3D_smooth 2 \
# 	            --save_tif --save_flow --save_outlines --in_folders \
# 	            --savedir $home/$sample/results/cpsam/output/$s/ \
# 	            --use_gpu --verbose
#             done
#     else
#         for q in {0..3}; do
# 	        python -m cellpose \
# 	        --image_path $home/$sample/processed/$SECTION/morphology/multi_layer/quatered/q0$q.ome.tif \
# 	        --do_3D --flow3D_smooth 2 \
# 	        --save_tif --save_flow --save_outlines --in_folders \
# 	        --savedir $home/$sample/results/cpsam/output/$SECTION/ \
# 	        --use_gpu --verbose
#         done
# fi