#!/bin/bash
#
echo $PWD
. get_toml_value.sh
CONFIG_FILE=${1-config.toml}
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
echo $home