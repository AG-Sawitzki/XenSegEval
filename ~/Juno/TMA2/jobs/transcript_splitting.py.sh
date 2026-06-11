#!/bin/bash
#
#SBATCH --job-name=transcript_splitting.py
#SBATCH --wait
#SBATCH --time=2-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --output=~/Juno/TMA2/run/logs/transcript_splitting.py_%N_%j.out
#SBATCH --error=~/Juno/TMA2/run/logs/transcript_splitting.py_%N_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50
#SBATCH --mail-user=julius.normann@bih-charite.de
#
. ~/.bashrc

export PIXI_CACHE_DIR=~/scratch/.cache/pixi#
pixi run python XenSegEval/processing/transcript_splitting.py
