#!/bin/bash
#
echo "$PWD"
. ./get_toml_value.sh
CONFIG_FILE=${1-config.toml}
SECTION=$2
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "name")
mm_path=$(get_toml_value "$CONFIG_FILE" "paths" "mm_path")
n_roi=$(get_toml_values "$CONFIG_FILE" "preprocessing" "n_roi")
#
. ~/.bashrc
#
if [ -z {$SECTION} ]
    then
        for s in {0..$((n_roi-1))}; do
            proseg \
            --xenium \
            --overwrite \
            --output-spatialdata $home/$sample/results/proseg/output/$s/spatialdata.zarr \
            --output-cell-polygons $home/$sample/results/proseg/output/$s/cell-polygons.geojson.gz \
            --output-cell-polygon-layers $home/$sample/results/proseg/output/$s/cell-polygons_layers.geojson.gz \
            --output-counts $home/$sample/results/proseg/output/$s/counts.mtx.gz \
            $home/$sample/processed/$SECTION/transcripts/relative.parquet
    else
        proseg \
        --xenium \
        --overwrite \
        --output-spatialdata $home/$sample/results/proseg/output/$SECTION/spatialdata.zarr \
        --output-cell-polygons $home/$sample/results/proseg/output/$SECTION/cell-polygons.geojson.gz \
        --output-cell-polygon-layers $home/$sample/results/proseg/output/$SECTION/cell-polygons_layers.geojson.gz \
        --output-counts $home/$sample/results/proseg/output/$SECTION/counts.mtx.gz \
        $home/$sample/processed/$SECTION/transcripts/relative.parquet
fi