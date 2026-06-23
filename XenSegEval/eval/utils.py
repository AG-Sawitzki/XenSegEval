# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    # get_false_negatives,
    # get_splits_and_merges
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


def wrapper_cs(mask, gt):
    gt_x = np.expand_dims(gt, axis=0)
    mask_x = np.expand_dims(mask, axis=0)

    gt_x_rl = relabel_sequential(gt_x)
    mask_x_rl = relabel_sequential(mask_x)

    pm = Metrics(method)

    object_metrics = pm.calc_object_stats(gt_x_rl, mask_x_rl)

    results = pd.DataFrame(data=object_metrics)

    return results


def wrapper_u4n(mask, gt):
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
    results = compute_af1_results(
        gt_rl,
        mask_rl,
        results,
        method
    )

    return results


def eval_masks(
    masks: list,
    gt: Union[str, os.PathLike, PosixPath],
    cs: ArrayLike,
    u4n: ArrayLike
) -> None:
    if type(gt) in [str, os.PathLike, PosixPath]:
        gt = tifffile.imread(gt)
    else:
        assert type(gt) is np.ndarray

    cs_arr = np.zeros((len(masks),))
    u4n_arr = np.zeros((len(masks),))
    for mask in masks:
        cs_results = wrapper_cs(mask, gt)
        u4n_results = wrapper_u4n(mask, gt)

        np.append(
            cs_arr,
            cs_results[
                ['precision', 'recall', 'f1', 'seg', 'jaccard', 'dice', 'PQ']
            ]
        )
        np.append(u4n_arr, u4n_results[['Threshold', 'F1', 'Jaccard']])

    np.append(cs, cs_arr)
    np.append(u4n, u4n_arr)

    return None


def cross_eval(
    results,
    methods,
    section,
) -> None:
    tested = []
    cs = np.array([])
    u4n = np.array([])
    for method in methods:
        tested.append(method)
        masks = list(
            Path(
                f'{results}/{method}/output/{section}/'
            ).glob('prediction*.tif')
        )
        gts = [
            list(Path(
                f'{results}/{to_test}/output/{section}/'
            ).glob('prediction*.tif')) for to_test in tested
        ]
        # for mask in masks:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(functools.partial(
                eval_masks,
                masks=masks,
                cs=cs,
                u4n=u4n
            ), gts)
            pool.close()
            pool.join()
    print(cs)
    print(u4n)
    # fig, ax = plt.subplots()
    # im, cbar = heatmap()
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
