# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)
# for cs-bench
from XenSegEval.eval.cs_benchmark.metrics import Metrics
# for plotting
from XenSegEval.plot import heatmap, annotate_heatmap

from skimage.segmentation import relabel_sequential
from skimage.morphology import label

import os
import gzip
import pickle
import functools
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import numpy as np
import tomlkit
import tifffile
from shapely.geometry import Polygon
import geopandas as gpd
import cv2

# types
from geopandas.geodataframe import GeoDataFrame
from numpy.typing import ArrayLike
from typing import Any, Union
from pathlib import PosixPath


def wrapper_cs(
    mask: ArrayLike,
    gt: ArrayLike,
    method: str = 'cross',
    outdir: Union[str, os.PathLike, PosixPath] = ''
) -> pd.core.frame.DataFrame:
    '''Wrapper for cs-benchmark. See [13] in README.md.
    Args:
        mask: Prediction to test against Ground Truth.
        gt: Ground Truth to test Prediction on.
        method (optional):
            Method name that is evaluated.
            Determines filename of json. See cs-benchmark docs.
        outdir (optional): Output directory to save json under. See above.
    Retruns:
        object_metrics in DataFrame.
    '''
    gt = np.squeeze(gt)
    mask = np.squeeze(mask)

    assert gt.shape == mask.shape, 'Mask and GT differ in shape.'

    gt_x = np.expand_dims(gt, axis=0)
    mask_x = np.expand_dims(mask, axis=0)

    pm = Metrics(method, outdir=outdir)

    object_metrics = pm.calc_object_stats(gt_x, mask_x)

    results = pd.DataFrame(data=object_metrics)

    return results


def wrapper_u4n(
    mask: ArrayLike,
    gt: ArrayLike,
    method='cross'
) -> tuple[
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
    pd.core.frame.DataFrame,
]:
    '''Wrapper for carpenterlab's evalutaion. See [12] in README.md.
    Args:
        mask: Prediction to test against Ground Truth.
        gt: Ground Truth to test Prediction on.
        method (optional):
            Method name that is evaluated.
            Appears in rows of results.
    Returns:
        df of metrics,
        df of false negatives,
        df of split merges
    '''
    mask = np.squeeze(mask)
    gt = np.squeeze(mask)

    assert gt.shape == mask.shape, 'Mask and GT differ in shape.'

    mask_l = label(mask)
    gt_l = label(gt)

    mask_rl = relabel_sequential(mask_l)[0]
    gt_rl = relabel_sequential(gt_l)[0]

    results = pd.DataFrame(
        columns=[
            'Method', 'Threshold', 'F1',
            'Jaccard', 'TP', 'FP', 'FN'
        ]
    )
    false_negatives = pd.DataFrame(
        columns=['False_Negative', 'Area']
    )
    split_merges = pd.DataFrame(
        columns=['Method', 'Merges', 'Splits']
    )
    results = compute_af1_results(
        gt_rl,
        mask_rl,
        results,
        method
    )

    false_negatives = get_false_negatives(
        gt_rl,
        mask_rl,
        false_negatives,
        method
    )

    split_merges = get_splits_and_merges(
        gt_rl,
        mask_rl,
        split_merges,
        method
    )

    return results, false_negatives, split_merges


def eval_mask(
    gt: Union[str, os.PathLike, PosixPath, ArrayLike],
    masks: list,
    cs: ArrayLike,
    u4n: ArrayLike,
    cs_val: str = 'f1',
    u4n_val: str = 'F1',
    threshold: int = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    '''Evaluate a single mask agains all other masks.
    Args:
        mask:
            Path to or Array of prediction to test.
            Functions as Prediction.
        gts:
            list of Paths to and/or Arrays of predictions.
            Function as Ground Truths
        cs: Array the cs_val is appended to.
        u4n: Array the u4n_val is appended to.
        cs_val (default: "f1"):
            one metric from [
                "f1", "seg", "jaccard", "dice", "PQ"
            ]
        u4n_val (default: "F1"):
            one metric from [
                "F1", "Jaccard"
            ]
        threshold (defaults: 0.5): Threshold for u4n. elem(0.5, 0.95)
    Retruns:
        cs and u4n
    '''
    if type(gt) in [str, os.PathLike, PosixPath]:
        gt = tifffile.imread(gt)
    assert type(gt) is np.ndarray, 'not an array'
    assert gt.shape == (1250, 1650), 'Wrong Shape'
    assert len(masks) > 0, 'No masks :<'
    # if type(mask) in [str, os.PathLike, PosixPath]:
    #     mask = tifffile.imread(mask)
    # assert type(mask) is np.ndarray
    # assert mask.shape == (1250, 1650)
    # assert len(gt) > 0, 'No masks :<'

    cs_arr = np.array([])
    u4n_arr = np.array([])

    for mask in masks:
        if type(mask) in [str, os.PathLike, PosixPath]:
            mask = tifffile.imread(mask)
        mask = np.squeeze(mask)
        assert mask.shape == (1250, 1650), 'Wrong Shape'

        cs_results = wrapper_cs(mask, gt)
        u4n_results, _, __ = wrapper_u4n(mask, gt)

        cs_arr = np.append(cs_arr, cs_results[cs_val])

        u4n_arr = np.append(
            u4n_arr,
            u4n_results[u4n_results['Threshold'] == threshold][u4n_val]
        )

    # if cs.shape == (0,):
    cs = np.append(cs, cs_arr)
    # else:
    #     cs = np.vstack([cs, cs_arr])
    # if u4n.shape == (0,):
    u4n = np.append(u4n, u4n_arr)
    # else:
    #     u4n = np.vstack([u4n, u4n_arr])

    return cs, u4n


def cross_eval(
    results: Union[str, os.PathLike, PosixPath],
    run: Union[str, os.PathLike, PosixPath],
    methods: list,
    section: str,
    threshold: int = 0.5,
) -> None:
    '''Evaluate each mask against every other.
    Args:
        results: path to directory containing all results.
        run: path to directory for run metrics and logs.
        methdos: list of all methods used for segmentation.
        section: string of section evaluation is running on.
        threshold (defaults: 0.5): Threshold for u4n. elem(0.5, 0.95)
    Returns:
        None
    '''
    cs = np.array([])
    u4n = np.array([])
    gts = [
        tifffile.imread(path) if method != 'mesmer'
        else tifffile.imread(path)[0, ...].squeeze()
        for method in methods for path in Path(
            f'{results}/{method}/output/{section}/'
        ).glob('prediction*.tif')
    ]
    for method in methods:
        if method == 'mesmer':
            masks = [
                tifffile.imread(path)[0, ...] for path in Path(
                    f'{results}/{method}/output/{section}/'
                ).glob('prediction*.tif')
            ]
        elif method == 'dissect':
            continue
        else:
            masks = [
                path for path in Path(
                    f'{results}/{method}/output/{section}/'
                ).glob('prediction*.tif')
            ]
        with open(f'{run}/eval_order.txt', 'a') as file:
            for path in masks:
                if type(path) is not np.ndarray:
                    file.writelines(f'{path}\n')
                else:
                    file.writelines(f'{method}\n')

        # if len(masks) < mp.cpu_count():
        #     if len(masks) == 0:
        #         print(method)
        #         continue
        #     else:
        #         processes = len(masks)
        # else:
        #     processes = mp.cpu_count()

        with mp.Pool(processes=mp.cpu_count()) as pool:
            res = pool.map(functools.partial(
                eval_mask,
                masks=masks,
                cs=cs,
                u4n=u4n,
                threshold=threshold
            ), gts)
            pool.close()
            pool.join()
    print(res)
    cs = np.vstack(res[:, 0])
    u4n = np.vstack(res[:, 1])
    print(cs)
    print(u4n)
    # fig, ax = plt.subplots()
    # im, cbar = heatmap()
    np.save(f'{results}/cs_cross.npy', cs)
    np.save(f'{results}/u4n_cross.npy', u4n)
    return None


def check_colour(
    r: int,
    g: int,
    b: int
) -> tuple:
    '''Gives a new rgb colour-tuple, incremented by 1.
    Args:
        r: red,
        g: green,
        b: blue
    Returns:
        Tuple of (r,g,b)
    '''
    if r < 255:
        r += 1
    else:
        if g < 255:
            g += 1
            r = 0
        else:
            if b < 255:
                b += 1
                g = 0
                r = 0
            else:
                return None
    return (r, g, b)


def polygon_to_mask(
    gdf: GeoDataFrame,
    shape: tuple,
    layer: int,
) -> ArrayLike:
    '''GeoJson Polygons to masks in a TIF.
    Args:
        gdf: path to geojson(.gz) or geodataframe.
        output_path: path to output location. might not be necessary.
    Retruns:
        Masks in numpy-array.
    '''
    r, g, b = (0,)*3
    img = np.zeros(shape, np.uint8)
    if type(layer) is int:
        gds = gdf[gdf['layer'] == layer]['geometry']
    else:
        gds = gdf['geometry']
    for mpg in gds:
        for lr in mpg.geoms:
            pl = np.array(list(lr.exterior.coords))
            cv2.fillPoly(img, np.int32([pl]), (r, g, b))
            r, g, b = check_colour(r, g, b)
    return img


def prepare_ProSeg(
    polygons: Union[str, os.PathLike, GeoDataFrame],
    output_path: Union[str, os.PathLike],
    shape: tuple
) -> None:
    '''A wrapper for polygon_to_mask.
    Args:
        polygons: path to the geojson file or GeoDataFrame.
        output_path: path to the dir to save the masks under.
        shape: shape of the corresponding groundtruth or known area shape.
    Returns:
        None. Saves masks as .tif in output_dir.
    '''
    if Path(polygons).suffix == '.gz':
        with gzip.open(polygons) as file:
            gdf = gpd.read_file(file)
    elif Path(polygons).suffix == 'geojson':
        gdf = gpd.read_file(polygons)
    elif type(polygons) is GeoDataFrame:
        gdf = polygons
    else:
        print('gdf not path or GeoDataFrame.')

    try:
        layers = max(gdf['layer'])
        for layer in range(layers+1):
            mask = polygon_to_mask(gdf, shape, layer)
            tf.imwrite(
                output_path / f'prediction_l{layer}.tif',
                mask
            )
    except KeyError:
        mask = polygon_to_mask(gdf, shape, layer=None)
        tf.imwrite(
            output_path / f'prediction.tif',
            mask
        )


# function form cellpose.utils
def outlines_list(masks, multiprocessing_threshold=1000, multiprocessing=None):
    '''Get outlines of masks as a list to loop over for plotting.
    Args:
        masks (ndarray): Array of masks.
        multiprocessing_threshold (int, optional):
            Threshold for enabling
            multiprocessing. Defaults to 1000.
        multiprocessing (bool, optional):
            Flag to enable multiprocessing. Defaults to None.
    Returns:
        list: List of outlines.
    Raises:
        None
    Notes:
        - This function is a wrapper for outlines_list_single and
          outlines_list_multi.
        - Multiprocessing is disabled for Windows.
    '''
    # default to use multiprocessing if not few_masks,
    # but allow user to override
    if multiprocessing is None:
        few_masks = np.max(masks) < multiprocessing_threshold
        multiprocessing = not few_masks
    # disable multiprocessing for Windows
    if os.name == "nt":
        if multiprocessing:
            logging.getLogger(__name__).warning(
                "Multiprocessing is disabled for Windows")
        multiprocessing = False
    if multiprocessing:
        print('  - Using Multiprocessing')
        return outlines_list_multi(masks)
    else:
        return outlines_list_single(masks)


# function form cellpose.utils
def outlines_list_single(masks):
    '''Get outlines of masks as a list to loop over for plotting.
    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)
    Returns:
        list: List of outlines as pixel coordinates.

    '''
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(
                mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix


# function form cellpose.utils
def outlines_list_multi(masks, num_processes=None):
    '''Get outlines of masks as a list to loop over for plotting.
    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)
    Returns:
        list: List of outlines as pixel coordinates.
    '''
    if num_processes is None:
        num_processes = cpu_count()
    unique_masks = np.unique(masks)[1:]
    with Pool(processes=num_processes) as pool:
        outpix = pool.map(
            get_outline_multi,
            [(masks, n) for n in unique_masks]
        )
    return outpix


# function form cellpose.utils
def get_outline_multi(args):
    '''Get the outline of a specific mask in a multi-mask image.
    Args:
        args (tuple): A tuple containing the masks and the mask number.
    Returns:
        numpy.ndarray: The outline of the specified mask as an array
                       of coordinates.

    '''
    masks, n = args
    mn = masks == n
    if mn.sum() > 0:
        contours = cv2.findContours(
            mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE
        )
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        return pix if len(pix) > 4 else np.zeros((0, 2))
    return np.zeros((0, 2))


# function form stackoverflow
# adapted to return shapely Polygons
def process_roi(npy_data, output_path):
    '''Mask to Polygons in GeoDataFrame (geojson) using Cellpose.utils.
    Args:
        npy_data: The numpy.ndarray of the masks.
        npy_base_output_path: Path to save the geojson.
    Returns:
        Nothing. Automatically saves the GDF.
    '''
    print(' - Extracting ROI')
    try:
        masks = npy_data.item().get("masks")
    except AttributeError:
        masks = npy_data
    masks = masks.squeeze()
    # change the index order:
    # first the cell then the layer it is on.
    # thus one would now how the same cell looks on different layers
    data = {'layer': [], 'name': [], 'geometry': []}
    if masks.ndim == 3:
        for z in range(masks.shape[0]):
            print(f' - Layer {z}')
            coords_list = outlines_list(masks[z, :, :])
            i = 1
            for coords in coords_list:
                data['layer'].append(z)
                data['name'].append(f'cell_{i}')
                data['geometry'].append(Polygon(coords))
                i += 1
    else:
        coords_list = outlines_list(masks)
        i = 1
        for coords in coords_list:
            data['layer'].append(np.nan)
            data['name'].append(f'cell_{i}')
            data['geometry'].append(Polygon(coords))
            i += 1
    gdf = gpd.GeoDataFrame(data=data)
    gdf.set_index(['layer', 'name'])
    print(' - Saving GeoDataFrame')
    gdf.to_file(output_path, driver='GeoJSON', index=True)
