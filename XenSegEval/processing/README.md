### `find_sections.py`
As mentioned, each function of the script can be run with user defined regions, but if those are not provided for the preprocessing and segmentation, then run this script to find the tissue-samples and number them 0-_n<sub>roi<sub>_ from top left to bottom right. <br>
The script tries to use the lowest subresolution of the `morphology.ome.tif`.

### `image_splitting.py`
As seen in the [Overview](#overview---segmentation-algorithms) most algorithms use the `morphology.ome.tif` or `morpholog_focus.ome.tif` image. To prepare them for the segmentation run `image_splitting.py`, which crops to the defined regions, with a default margin of 1% of the ROI's size. Additionally the regions are also saved as single-layer images, and both, multi-layer and single-layer, are split into `chunks` if a value above 0 is provided.

### `transcript_splitting.py`
For the "SRT based" and "Mixed" algorithms the `transcript.parquet` is seperated into smaller tables containing only one section each. The process uses pythons multiprocessing and polars filter functions.

### `boundaries_splitting.py`
Xenium provides boundaries of cells and nuclei, saved in the `*_boundaries.parquet` files. They are seperated into smaller tables just like the SRT data.