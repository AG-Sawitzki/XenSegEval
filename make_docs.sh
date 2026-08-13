#!/bin/bash
# 
# docs for main
pixi run pdoc ./XenSegEval/main.py -o ./docs/ --docformat=numpy
# 
# docs for utils
pixi run pdoc ./XenSegEval/utils.py -o ./docs/ --docformat=numpy
# 
# docs for processing
pixi run pdoc ./XenSegEval/processing/ -o ./docs/ --docformat=numpy
# 
# docs for eval
pixi run -e eval pdoc ./XenSegEval/eval/utils.py -o ./docs/ --docformat=numpy
## docs for cross
pixi run -e eval pdoc ./XenSegEval/eval/cross/cross.py -o ./docs/ --docformat=numpy
## docs for free
pixi run -e free pdoc ./XenSegEval/eval/free/free.py -o ./docs/ --docformat=numpy
## docs for masked
pixi run -e eval pdoc ./XenSegEval/eval/masked/eval.py -o ./docs/ --docformat=numpy
