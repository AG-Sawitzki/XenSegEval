#!/bin/bash
#
echo $PWD
. get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
sections_path=$(get_toml_value "config.toml" "paths" "sections_path")
#
. ~/.bashrc
echo $home
echo $sections_path
sections=$(jq 'keys' $sections_path | jq .[] | tr -d ' "')
#
echo $sections
for s in ${sections}; do
        echo $home/$sample/processed/$s/transcripts/relative.parquet
done
