#!/bin/bash
#
echo "$PWD"
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
sections_path=$(get_toml_values "$CONFIG_FILE" "paths" "sections")
n_roi=$(get_toml_values "$CONFIG_FILE" "preprocessing" "n_roi")
#
. ~/.bashrc
sections=$(jq 'keys' $sections_path | jq .[] | tr -d ' "')
#
for s in ${sections}; do
    proseg \
    --xenium \
    --overwrite \
    --output-spatialdata $home/$sample/results/proseg/output/$s/spatialdata.zarr \
    --output-cell-polygons $home/$sample/results/proseg/output/$s/cell-polygons.geojson.gz \
    --output-cell-polygon-layers $home/$sample/results/proseg/output/$s/cell-polygons_layers.geojson.gz \
    --output-counts $home/$sample/results/proseg/output/$s/counts.mtx.gz \
    $home/$sample/processed/$SECTION/transcripts/relative.parquet