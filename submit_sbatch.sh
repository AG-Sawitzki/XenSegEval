#!/bin/bash
#
cd /data/cephfs-1/work/groups/sawitzki/users/juno12_c/10xSegEval/
. ./get_toml_value.sh
CONFIG_FILE=${2-config.toml}
SECTION=$3
home=$(get_toml_value "$CONFIG_FILE" "paths" "home")
sample=$(get_toml_value "$CONFIG_FILE" "paths" "sample_name")
mail=$(get_toml_value "$CONFIG_FILE" "paths" "mail")
#
IN=$1
IFS='/' read -r -a array <<< "$IN"
name=${array[-1]}
#
sbatch <<EOT
#!/bin/bash
#
#SBATCH --job-name=""$name
#SBATCH --gres=gpu:l40:1
#SBATCH --time=2-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=$home/$sample/run/$method/logs/$name/%N_%j.out
#SBATCH --error=$home/$sample/run/$method/logs/$name/%N_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50
#SBATCH --mail-user=$mail
#
if [ "$IN" == "*.sh" ]
    then
        source $IN $CONFIG_FILE $SECTION
    else
        python $IN -c $CONFIG_FILE -s $SECTION
fi
EOT