#!/bin/bash

source get_toml_value.sh
CONFIG_FILE=$1
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
bashrc_path=$(get_toml_value "$CONFIG_FILE" "paths" "bashrc_path")
CHECK_GENE_MAP=$(get_toml_value "$CONFIG_FILE" "methods.ucs" "check_gene_map")
source $bashrc_path
micromamba activate ucs

cd /data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/scripts/segmenting/seg_UCS

# Run the preprocess to get the gene map and official nuclei mask
python preprocess/xenium.py \
--transcripts $home/$sample/processed/$SECTION/transcripts/relative.csv \ 
--cell_boundary_10X $home/$sample/processed/boundaries/cell_relative.parquet \
--nucleus_boundary_10X $home/$sample/processed/boundaries/nucleus_relative.parquet \
--out_dir $home/$sample/results/ucs/$SECTION

if [ CHECK_GENE_MAP == True ]
    then
        python preprocess/check_paired.py \
        --gene_map $home/$sample/results/ucs/$SECTION/gene_map.tif \
        --segmentation $home/$sample/results/ucs/$SECTION/nuclei_10X_mask.tif \
        --region 1000 2000 1000 2000 \
        --out_dir $home/$sample/results/ucs/$SECTION
fi

# Run UCS
python run.py \
--gene_map $home/$sample/results/ucs/$SECTION/gene_map.tif \
--nuclei_mask $home/$sample/results/ucs/$SECTION/nuclei_10X_mask.tif \
--log_dir $home/$sample/run/ucs/$SECTION/log/