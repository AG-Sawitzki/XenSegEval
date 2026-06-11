# XenSegEval
Segments on Xenium v2 output and evaluates the results if a ground-truth is provided.

$$\color{orange}\text{A Linux-64 system is currently strictly necessary!}$$ <br>
$$\color{orange}\text{The automatic pipeline requires to be run on the BIH-HPC cluster!}$$ <br>

## ToDo
- [ ] make chunks optional
- [ ] make single-layer optional
- [ ] make boundaries optional (only used by UCS)
- [ ] doc-strings
- [X] proseg in pixi.toml
- [ ] add PCA
- [ ] add PD
- [ ] add sources
    - [ ] CPSAM
    - [ ] DeepCell
    - [ ] DINOCell
    - [ ] DISSECT
    - [ ] Proseg
    - [ ] StarDist
    - [ ] UCS
    - [ ] Evaluation
    - [ ] Xenium by 10xGenomics

## Overview - Segmentation Algorithms

| Training | SRT based | Image based | Mixed |
| --- | --- | --- | --- |  
| pre-trained |  | $$\color{green}\text{CellposeSAM}$$ <br> $$\color{green}\text{StarDist}$$ <br> $$\color{green}\text{Mesmer}$$ <br> $$\color{green}\text{DINOCell}$$ | $$\color{red}\text{SCS}$$ <br> $$\color{green}\text{DISSECT}$$ |
| un-trained |  |  | $$\color{red}\text{segger}$$ <br> $$\color{green}\text{UCS}$$ |
| no-training | $$\color{green}\text{Proseg}$$ |  | $$\color{red}\text{ComSeg}$$ <br> $$\color{red}\text{RNA2Seg}$$ |

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
in the directory to update the packages.

### Dissect
To install dissect-st run the following commads
```
pixi shell -e deepcell
uv pip install --prerelease=allow deterctron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

### Proseg
Install Proseg by running the commands below
```
pixi shell -e proseg
cargo install proseg
```

# Pipeline
This repository can prepare and segment on Xenium v2, and soon v3, output. If a ground-truth is provided, it evaluates the results using basic Jaccard values. [PCA](https://github.com/murphygroup/CellSegmentationEvaluator) or [probability density](https://github.com/lstrgar/seg) based evaluation are in work.

The preprocessing steps can be performed without pre-defined ROIs, but **for the evaluation a json file with coordinates** must be provided! Structured as below.
```
{"name":
    [[y0, x0], [y1, x1]]
}
```
Where `y0 & x0` define the top left corner and `y1 & x1` the bottom right corner in pixel! Add the path to this json file to the `config.toml` under `[paths]`.

Under `[paths]` the scripts find the path to the raw data, the name of the sample, and, if needed, the path to the directory the output should be saved under.

After configuring the paths in `config.toml` you can start the pipeline using
```
pixi shell
python XenSegEval/main.py
```

## Preprocessing
...
### `find_sections.py`
As mentioned each function of the script can be run with user defined regions, but if those are not providied for the preprocessing and segmentation, then run this script to find the tissue-samples and number them 0-_n<sub>roi<sub>_ from top left to bottom right. <br>
The script tries to use the lowest subresolution of the `morphology.ome.tif`.

### `image_splitting.py`
As seen in the [Overview](#overview---segmentation-algorithms) most algorithms use the `morphology.ome.tif` or `morpholog_focus.ome.tif` image. To prepare them for the segmentation run `image_splitting.py`, which crops to the defined regions, with a default margin of 1% of the ROI's size. Additionally the regions are also saved as single-layer images, and both, multi-layer and single-layer, are by default split into 4 chunks.

### `transcript_splitting.py`
For the "SRT based" and "Mixed" algorithms the `transcript.csv.gz` (later `transcripts.parquet`) is seperated into smaller tables containing only one section each. The process uses pythons multiprocessing.

### `boundaries_splitting.py`
Xenium provides boundaries of cells and nuclei, saved in the `_boundaries.parquet` files. They are seperated into smaller tables just like the SRT data.

## Segmenting
...
## Evaluating
...
