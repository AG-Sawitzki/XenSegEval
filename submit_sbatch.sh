#!/bin/bash
#
source /path/to/lib_ini.sh
CONFIG_FILE=$1
home=$(ini_read "$CONFIG_FILE" "PATHS" "home")
sample=$(ini_read "$CONFIG_FILE" "PATHS" "sample_name")
mail=$(ini_read "$CONFIG_FILE" "PATHS" "mail)
#
IN=$1
IFS='/' read -r -a array <<< "$IN"
echo "${array[-1]}"
name=${array[-1]}
#
method=$2
sbatch <<EOT
#!/bin/bash
#
#SBATCH --job-name=""$name
#SBATCH --gres=gpu:l40:1
#SBATCH --time=2-00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --output=$home/$sample/run/$method/logs/$name/%N_%j.out
#SBATCH --error=$home/$sample/run/$method/logs/$name/%N_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50
#SBATCH --mail-user=$mail
#

source $1

EOT