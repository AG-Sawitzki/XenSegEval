#!/bin/bash
#
. get_toml_value.sh
CONFIG_FILE=$1
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
n_roi=$(get_toml_value "$CONFIG_FILE" "preprocessing" "n_roi")
sections_path=$(get_toml_value "$CONFIG_FILE" "paths" "sections_path")
CHECK_GENE_MAP=$(get_toml_value "$CONFIG_FILE" "methods.ucs" "check_gene_map")
#
. ~/.bashrc
sections=$(jq 'keys' $sections_path | jq .[] | tr -d ' "')
#pixi shell -e ucs
#
echo $sections
for s in ${sections}; do
    echo $s
    # Run the preprocess to get the gene map and official nuclei mask
    python -m XenSegEval.segmenting.ucs.xenium \
    --transcripts $home/$sample/processed/$s/transcripts/relative.parquet \ 
    --cell_boundary_10X $home/$sample/processed/boundaries/cell_relative.parquet \
    --nucleus_boundary_10X $home/$sample/processed/boundaries/nucleus_relative.parquet \
    --out_dir $home/$sample/results/ucs/$s/
    #
    if [ CHECK_GENE_MAP == true ]
        then
             python preprocess/check_paired.py \
             --gene_map $home/$sample/results/ucs/$s/gene_map.tif \
             --segmentation $home/$sample/results/ucs/$s/nuclei_10X_mask.tif \
             --region 1000 2000 1000 2000 \
             --out_dir $home/$sample/results/ucs/$SECTION
    fi
    #
    # Run UCS
    python -m run \
    --gene_map $home/$sample/results/ucs/$s/gene_map.tif \
    --nuclei_mask $home/$sample/results/ucs/$s/nuclei_10X_mask.tif \
    --log_dir $home/$sample/run/ucs/log/$s/
done
