# 10xSegEval
Segments on Xenium v2 output and Evaluates the results if a ground-truth is provided.

### Warning 
```diff
- A Linux-64 system is currently strictly necessary!  
- The automatic pipeline requires to be run on the BIH-HPC cluser!
```
### Overview - Segmentation Algorithms

| Training | SRT based | Image based | Mixed |
| --- | --- | --- | --- |  
| pre-trained |  | $$\color{green}\text{CellposeSAM}$$ $$\color{green}\text{StarDist}$$ $$\color{green}\text{Mesmer}$$ $$\color{green}\text{DinoCell}$$ | $$\color{red}\text{SCS}$$ $$\color{green}\text{DISSECT}$$ |
| un-trained |  |  | $$\color{red}\text{segger}$$ $$\color{green}\text{UCS}$$ |
| no-training | $$\color{green}\text{Proseg}$$ |  | $$\color{red}\text{ComSeg}$$ $$\color{red}\text{RNA2Seg}$$ |

All those in green are currently working. Those in red have been tried and were either uninstallable (segger & SCS) or could not work with the data (ComSeg & RNA2Seg).  
"SRT based" includes those that require only the transcriptomics data.  
"Image based" means the input contains multi-layer or single-layer (tiff) images.  
"Mixed" are those which require both, image and transcript location.

# Getting Started
To use this repository, clone it
```
git clone https://github.com/Normann-BPh/10xSegEval.git
```
install pixi
```
curl -fsSL https://pixi.sh/install.sh | sh
# or
wget -q0- https://pixi.sh/install.sh | sh
```
then run
```
pixi lock
```
to update the packages.

## Proseg
For now Proseg is, as a rust package, installed using the `install_proseg.sh` file. This should change in the future and be included in one of the pixi environments.

# Pipeline
This repository can prepare and segment on Xenium v2, and soon v3, output, and, if a ground-truth is provided, evaluate the results using basic Jaccard values. [PCA](https://github.com/murphygroup/CellSegmentationEvaluator) or [probability density](https://github.com/lstrgar/seg) based evaluation are in work.

The preprocessing steps can be performed with out pre-defined ROIs, but **for the evaluation a json file with coordinates** must be provided! Structured as below.
```
{"name":
    [[y0, x0], [y1, x1]]
}
```
Where `y0 & x0` define the top left corner and `y1 & x1` the bottom right corner.

## Preprocessing

As seen in the [Overview](#overview---segmentation-algorithms) most algorithms use the `morphology.ome.tif` or `morpholog_focus.ome.tif` image.

# proseg
start/proseg.sh has the flag --overwrite (L18) active. if you do not want old files to be overwritten then remove this line.