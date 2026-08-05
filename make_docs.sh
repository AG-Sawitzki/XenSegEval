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
# docs for segmenting
## docs for CPSAM
# pixi run -e cpsam pdoc ./XenSegEval/segmenting/seg_cpsam.py -o ./docs/ --docformat=numpy
# ## docs for DINOCell
# pixi run -e dinocell pdoc ./XenSegEval/segmenting/seg_dinocell.py -o ./docs/ --docformat=numpy
# ## docs for DISSECT
# pixi run -e dissect pdoc ./XenSegEval/segmenting/seg_dissect.py -o ./docs/ --docformat=numpy
# ## docs for MESMER
# pixi run -e deepcell pdoc ./XenSegEval/segmenting/seg_mesmer.py -o ./docs/ --docformat=numpy
# ## docs for ProSeg
# pixi run -e proseg pdoc ./XenSegEval/segmenting/seg_proseg.py -o ./docs/ --docformat=numpy
# ## docs for StarDist
# pixi run -e stardist pdoc ./XenSegEval/segmenting/seg_stardist.py -o ./docs/ --docformat=numpy
# 
# docs for eval
pixi run -e eval pdoc ./XenSegEval/eval/utils.py -o ./docs/ --docformat=numpy
## docs for cross
pixi run -e eval pdoc ./XenSegEval/eval/cross/ -o ./docs/ --docformat=numpy
## docs for free
pixi run -e free pdoc ./XenSegEval/eval/free/ -o ./docs/ --docformat=numpy
## docs for masked
pixi run -e eval pdoc ./XenSegEval/eval/masked/ -o ./docs/ --docformat=numpy
