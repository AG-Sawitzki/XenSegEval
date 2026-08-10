#!/bin/bash
#
. ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64/:./.pixi/envs/segger/lib/
# segger segment -i $1 -o $2
#
# segger export boundaries -s $2/segger_segmentation.parquet -i $1 -o $2
pixi run python -m XenSegEval.segmenting.seg_segger -c $1