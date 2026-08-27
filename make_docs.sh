#!/bin/bash
# 
# not working :<
#
# docs for main
pixi run -e docs pdoc ./XenSegEval/main.py -o ./docs/ --docformat=numpy
# 
# docs for utils
pixi run -e docs pdoc ./XenSegEval/utils.py -o ./docs/ --docformat=numpy
# 
# docs for processing
pixi run -e docs pdoc ./XenSegEval/processing/ -o ./docs/ --docformat=numpy
# 
# docs for eval
pixi run -e docs pdoc ./XenSegEval/eval/utils.py -o ./docs/ --docformat=numpy
## docs for cross
pixi run -e docs pdoc ./XenSegEval/eval/cross/cross.py -o ./docs/ --docformat=numpy
## docs for free
pixi run -e docs pdoc ./XenSegEval/eval/free/free.py -o ./docs/ --docformat=numpy
## docs for masked
pixi run -e docs pdoc ./XenSegEval/eval/masked/eval.py -o ./docs/ --docformat=numpy
