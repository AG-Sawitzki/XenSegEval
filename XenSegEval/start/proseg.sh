#!/bin/bash
#
# echo $PWD
# . get_toml_value.sh
# CONFIG_FILE=${1-config.toml}
# home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
# sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
# sections_path=$(get_toml_value "$CONFIG_FILE" "paths" "sections_path")
#
. ~/.bashrc
pixi run python -m XenSegEval.segmenting.seg_proseg -c $1
#
# sections=$(jq 'keys' $sections_path | jq .[] | tr -d ' "')
#
# for s in ${sections}; do
#     mkdir -p "$home/$sample/results/proseg/output/$s/"
#     proseg \
#     --xenium \
#     --overwrite \
#     --output-spatialdata "$home/$sample/results/proseg/output/$s/spatialdata.zarr" \
#     --output-cell-polygons "$home/$sample/results/proseg/output/$s/cell-polygons.geojson.gz" \
#     --output-cell-polygon-layers "$home/$sample/results/proseg/output/$s/cell-polygons_layers.geojson.gz" \
#     --output-counts "$home/$sample/results/proseg/output/$s/counts.mtx.gz" \
#     "$home/$sample/processed/$s/transcripts/relative.csv.gz"
# done
