# XenSegEval
Segments on XeniumV2-output<sup>[[0]](#0)</sup> and evaluates<sup>[[12]](#12)[[13]](#13)</sup> the results if a ground-truth is provided.

> [!IMPORTANT]
> A Linux-64 system is currently strictly necessary!
> The automatic pipeline requires to be run on the BIH-HPC cluster!

## ToDo
- [X] license
- [X] make chunks optional
- [X] make single-layer optional
- [ ] make boundaries optional (only used by UCS)
- [X] doc-strings
- [X] proseg in pixi.toml
- [ ] add PCA
- [ ] add PD
    - [ ] and source
- [X] add sources
    - [X] CPSAM
    - [X] DeepCell
    - [X] DINOCell
    - [X] DISSECT
    - [X] Proseg
    - [X] StarDist
    - [X] UCS
    - [X] Xenium by 10xGenomics

## Overview - Segmentation Algorithms

| Training | SRT based | Image based | Mixed |
| --- | --- | --- | --- |  
| pre-trained |  | $$\color{green}\text{CellposeSAM}$$ <sup>[[3]](#3)</sup> <br> $$\color{green}\text{StarDist}$$ <sup>[[1]](#1)[[2]](#2)</sup> <br> $$\color{green}\text{Mesmer}$$ <sup>[[6]](#6)[[7]](#7)[[8]](#8)[[9]](#9)</sup> <br> $$\color{green}\text{DINOCell}$$ <sup>[[4]](#4)</sup> | $$\color{red}\text{SCS}$$ <br> $$\color{green}\text{DISSECT}$$ <sup>[[5]](#5)</sup> |
| un-trained |  |  | $$\color{red}\text{segger}$$ <br> $$\color{green}\text{UCS}$$ <sup>[[10]](#10)</sup> |
| no-training | $$\color{green}\text{Proseg}$$ <sup>[[11]](#11)</sup> |  | $$\color{red}\text{ComSeg}$$ <br> $$\color{red}\text{RNA2Seg}$$ |

All those in green are currently working. Those in red have been tried and were either uninstallable (segger & SCS) or could not work with the data (ComSeg & RNA2Seg).  
"SRT based" includes those that require only the transcriptomics data.  
"Image based" means the input contains multi-layer or single-layer (tiff) images.  
"Mixed" are those which require both, image and transcript location.

# Getting Started
To use this repository, clone it
```
git clone https://github.com/AG-Sawitzki/XenSegEval.git
```
install pixi
```
curl -fsSL https://pixi.sh/install.sh | sh
# or
wget -q0- https://pixi.sh/install.sh | sh
```
then change the directory and update the `pixi.lock` file.
```
cd XenSegEval
pixi lock
```

<!---### CellposeSAM/DINO
To use the DINO backend of Cellpose install the package with
```
pixi shell -e cpsam
uv pip install git+https://github.com/facebookresearch/dinov3
```
-->
### Dissect
<!--To install dissect-st run the following commads
```
pixi shell -e dissect
python3 -m ensurepip
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
<!-- uv pip install --prerelease=allow deterctron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html 
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
-->
(Afterwards) Download the pre-trained model using gdown (v5.2.2).
```
gdown --fuzzy 'https://drive.google.com/file/d/1Y9_YCJzhUPEQBDAdKVyrKplI1vpD4qiO/view?usp=sharing' -O XenSegEval/segmenting/dissect/dissect_weights.pth
```
The `config.yaml` file can be found on the [ZengLab GitHub](https://github.com/zenglab-pku/DISSECT/blob/main/config.yaml). After downloading the file add it to the same directory as the weights.

### Proseg
Install Proseg by running the command below
```
pixi run -e proseg cargo install proseg
```

# Pipeline
This repository can prepare and segment on Xenium v2, and soon v3, output. If a ground-truth is provided, it evaluates the results using basic Jaccard values. [PCA](https://github.com/murphygroup/CellSegmentationEvaluator) or [probability density](https://github.com/lstrgar/seg) based evaluation are in work.

The preprocessing steps can be performed without pre-defined ROIs, but **for the evaluation a json file with coordinates** must be provided. Structured as below.
```
{"name":
    [[y0, x0], [y1, x1]]
}
```
Where `y0 & x0` define the top left corner and `y1 & x1` the bottom right corner in pixel! Add the path to this json file to the `config.toml` under `[paths]` or include the `--Section` flag when starting `XenSegEval.main`.

Under `[paths]` the scripts find the path to the raw data, the name of the sample, and the path to the directory the output should be saved under.

After configuring the paths in `config.toml` you can start the pipeline using
```
pixi run python -m XenSegEval.main
```

## Preprocessing
Finds coordinates of ROIs if not provided. Splits morphology.ome.tif and morphology_focus.ome.tif, transcripts.csv, and boundaries.parquet accordingly.<br>
Run `main.py` with `[Tasks.preprocess]` set to `true` in the config.toml.<br>
Or run the scripts seperately.

## Segmenting
The Algorithms marked in [green](#overview---segmentation-algorithms) are started if they appear in the config.toml as
```
[methods.<method-name>]
# parameters
``` 
and `[Tasks.segment]` is set to `true`.<br>
Alternatively start them with their bash or python script.
## Evaluating
If `[Tasks.evaluate] = true` all those procedures set to `true` under `[evaluation]` will be used to evaluate the segmentation of all available segmentation methods.

# References
## Xenium
<a id="0">[0] 
Janesick, A. et al. (2023)<br>
High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis.<br>
[DOI:10.1038/s41467-023-43458-x](doi.org/10.1038/s41467-023-43458-x)

## Segmentation-Algorithms
<a id="1">[1]</a>
Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, Gene Myers (2020).<br>
Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.<br>
[DOI:10.1109/WACV45572.2020.9093435](doi.org/10.1109/WACV45572.2020.9093435)

<a id="2">[2]</a>
Martin Weigert, Uwe Schmidt (2022).<br>
Nuclei Instance Segmentation and Classification in Histopathology Images with Stardist.<br>
[DOI:10.1109/ISBIC56247.2022.9854534](doi.org/10.1109/ISBIC56247.2022.9854534)

<a id="3">[3]</a>
Marius Pachitariu, Michael Rariden, Carsen Stringer (2025). <br>
Cellpose-SAM: superhuman generalization for cellular segmentation.<br>
[DOI:10.1101/2025.04.28.651001](doi.org/10.1101/2025.04.28.651001)

<a id="4">[4]</a>
Kaden Stillwagon, Alexandra Dunnum VandeLoo, Benjamin Magondu, Craig R. Forest (2026).<br>
Self-supervised Pretraining of Cell Segmentation Models.<br>
[DOI:10.48550/arXiv.2604.10609](doi.org/10.48550/arXiv.2604.10609)

<a id="5">[5]</a>
Zeng Lab. <br>
DISSECT integrates cytological images and spatial transcriptomics for cell segmentation.

<a id="6">[6]</a>
David A. Van Valen, Takamasa Kudo, Keara M. Lane, Derek N. Macklin, Nicolas T. Quach, Mialy M. DeFelice, Inbal Maayan, Yu Tanouchi, Euan A. Ashley, Markus W. Covert (2016)<br>
Deep Learning Automates the Quantitative Analysis of Individual Cells in Live-Cell Imaging Experiments.<br>
[DOI:10.1371/journal.pcbi.1005177](doi.org/10.1371/journal.pcbi.1005177)

<a id="7">[7]</a>
Erick Moen, Enrico Borba, Geneva Miller, Morgan Schwartz, Dylan Bannon, Nora Koe, Isabella Camplisson, Daniel Kyme, Cole Pavelchek, Tyler Price, Takamasa Kudo, Edward Pao, William Graf, David Van Valen (2019)<br>
Accurate cell tracking and lineage construction in live-cell imaging experiments with deep learning.<br>
[DOI:10.1101/803205](doi.org/10.1101/803205)

<a id="8">[8]</a>
Bannon, D., Moen, E., Schwartz, M. et al. (2021)<br>
DeepCell Kiosk: scaling deep learning–enabled cellular image analysis with Kubernetes.<br>
[DOI:10.1038/s41592-020-01023-0](doi.org/10.1038/s41592-020-01023-0)

<a id="9">[9]</a>
Greenwald, N.F., Miller, G., Moen, E. et al. (2022)<br>
Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning.<br>
[DOI:10.1038/s41587-021-01094-0](doi.org/10.1038/s41587-021-01094-0)

<a id="10">[10]</a>
Yuheng Chen, Xin Xu, Xiaomeng Wan, Jiashun Xiao, Can Yang (2025)<br>
UCS: A Unified Approach to Cell Segmentation for Subcellular Spatial Transcriptomics.<br>
[DOI:10.1002/smtd.202400975](doi.org/10.1002/smtd.202400975)

<a id="11">[11]</a>
Jones, D.C., Elz, A.E., Hadadianpour, A. et al. (2025)<br>
Cell simulation as cell segmentation.<br>
[DOI:10.1038/s41592-025-02697-0](doi.org/10.1038/s41592-025-02697-0)

## Evaluation-Methods
<a id="12">[12]</a>
Caicedo, Juan C., et al. (2019)<br>
Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images.<br>
[DOI:10.1002/cyto.a.23863](doi.org/10.1002/cyto.a.23863)

<a id="13">[13]</a>
Can Shi, Jinghong Fan, Zhonghan Deng, Huanlin Liu, Qiang Kang, Yumei Li, Jing Guo, Jingwen Wang, Jinjiang Gong, Sha Liao, Ao Chen, Ying Zhang, Mei Li (2025)<br>
CellBinDB: a large-scale multimodal annotated dataset for cell segmentation with benchmarking of universal models.<br>
[DOI:10.1101/2024.11.20.619750](doi.org/10.1101/2024.11.20.619750)
