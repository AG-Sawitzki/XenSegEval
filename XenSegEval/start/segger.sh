#!/bin/bash
#
. ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64/:./.pixi/envs/segger/lib/
pixi run -e segger python -m XenSegEval.segmenting.seg_segger -c $1