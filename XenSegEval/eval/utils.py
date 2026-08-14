from XenSegEval.processing.utils import (
    wrap_ptm
)
# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)
# for cs-bench
from XenSegEval.eval.cs_benchmark.metrics import Metrics
# for plotting
# from XenSegEval.plotting.utils import heatmap, annotate_heatmap

from skimage.segmentation import relabel_sequential
from skimage.morphology import label

import os
import gzip
import json
import pickle
import functools
from pathlib import Path
import multiprocessing as mp

import polars as pl
import pyarrow as pa

import pandas as pd
import numpy as np
import tomlkit
import tifffile
from shapely.geometry import Polygon
import geopandas as gpd
import cv2
import matplotlib.pyplot as plt

# types
from numpy.typing import ArrayLike
from typing import Any, Union
from pathlib import PosixPath

TABLE = pa.lib.Table
GDF = gpd.geodataframe.GeoDataFrame
PDF = pl.dataframe.frame.DataFrame
DF = pd.DataFrame


def mean_cross_eval(
    arr,
    methods,
    labels,
):
    avg = []
    avgT = []
    for method in methods:
        indices = [
            i for i,x in enumerate(labels) if method in x
        ]
        for i in indices:
            a = arr[i]
            aT = arr.T[i]
            a = np.delete(a, indices, axis=0)
            aT = np.delete(aT, indices, axis=0)
            avg = np.append(avg, np.mean(a))
            avgT = np.append(avgT, np.mean(aT))
    avg = np.expand_dims(avg, axis=1)
    avgT = np.append(avgT, np.nan)
    avgT = np.expand_dims(avgT, axis=0)
    arr = np.hstack((arr, avg))
    arr = np.vstack((arr, avgT))
    return arr, avg, avgT


def wrapper_cs(
    dt: ArrayLike,
    gt: ArrayLike,
    # outdir: Union[str, os.PathLike, PosixPath],
    method: str = 'cross',
) -> pd.core.frame.DataFrame:
    '''Wrapper for cs-benchmark. See [13] in README.md.

    Parameters
    ----------
        mask : ArrayLike
            Prediction to test against Ground Truth.
        gt : ArrayLike
            Ground Truth to test Prediction on.
        outdir : Path, optional
            Output directory to save json under. See above.
        method : str, optional
            Method name that is evaluated.
            Determines filename of json. See cs-benchmark docs.
            Default is `cross`.
    Retruns:
    ----------
        out : DataFrame
            object_metrics in DataFrame.
        If `outdir` given then saved as json.
    '''
    print('starting')
    if type(dt) is list:
        # print('is list')
        dt_x = np.array([
            np.squeeze(i) for i in dt
        ])
        gt = np.squeeze(gt)
        gt_x = np.array([
            gt for i in range(len(dt))
        ])
    else:
        # print('is single')
        dt = np.squeeze(dt)
        dt_x = np.expand_dims(dt, axis=0)
        gt = np.squeeze(gt)
        gt_x = np.expand_dims(gt, axis=0)
        # assert gt.shape == dt.shape, 'DT and GT differ in shape.'

    gt_x = np.int64(gt_x)
    dt_x = np.int64(dt_x)
    # print('could convert')
    # outdir = Path(outdir)

    pm = Metrics(method)

    object_metrics = pm.calc_object_stats(gt_x, dt_x)
    # print('could calc')
    results = pd.DataFrame(data=object_metrics, dtype=float)

    # if outdir is not None:
    #     if Path(outdir / f'{method}_cs_all.csv').is_file():
    #         results.to_csv(
    #             outdir / f'{method}_cs_all.csv',
    #             mode='a', header=False, index=False
    #         )
    #     else:
    #         results.to_csv(
    #             outdir / f'{method}_cs_all.csv',
    #             index=False
    #         )

    return results


def wrapper_u4n(
    dt: ArrayLike,
    gt: ArrayLike,
    method: str = 'cross'
) -> tuple[GDF, GDF, GDF]:
    '''Wrapper for carpenterlab's evalutaion. See [12] in README.md.
    Parameters
    ----------
        dt : ArrayLike
            Prediction to test against Ground Truth.
        gt : ArrayLike
            Ground Truth to test Prediction on.
        method : str, otpional
            Method name that is evaluated.
            Appears in rows of results.
            Default is `cross`
    Returns:
    ----------
        out : tuple
            df of metrics,
            df of false negatives,
            df of split merges
    '''
    dt = np.squeeze(dt)
    gt = np.squeeze(gt)

    assert gt.shape == dt.shape, 'DT and GT differ in shape.'
    # print('are same shape')
    dt_l = label(dt)
    gt_l = label(gt)

    dt_rl = relabel_sequential(dt_l)[0]
    gt_rl = relabel_sequential(gt_l)[0]
    # print('relabled')
    results = pd.DataFrame(
        columns=[
            'Method', 'Threshold', 'F1',
            'Jaccard', 'TP', 'FP', 'FN'
        ],
        dtype=float
    )
    false_negatives = pd.DataFrame(
        columns=['False_Negative', 'Area'],
        dtype=float
    )
    split_merges = pd.DataFrame(
        columns=['Method', 'Merges', 'Splits'],
        dtype=float
    )
    results = compute_af1_results(
        gt_rl,
        dt_rl,
        results,
        method
    )

    false_negatives = get_false_negatives(
        gt_rl,
        dt_rl,
        false_negatives,
        method
    )

    split_merges = get_splits_and_merges(
        gt_rl,
        dt_rl,
        split_merges,
        method
    )
    # print('got all clacs done')
    return results, false_negatives, split_merges


def eval_mask(
    gt: Union[str, os.PathLike, PosixPath, ArrayLike],
    dts: list,
    arr: ArrayLike,
    metric: str = 'f1',
    benchmark: str = 'cs',
    threshold: float = 0.5,
) -> ArrayLike:
    '''Evaluate a single mask agains all other masks.
    Parameters
    ----------
        gt : Path
            Path to or Array of prediction to test.
            Functions as ground truth.
        dts : list
            list of Paths to and/or Arrays of predictions.
            Function as predictions.
        arr : ArrayLike
             Array the metric is appended to.
        metric : str, optional
            if benchmark = "cs":
                "f1" | "seg" | "jaccard" | "dice" | "PQ"
            if benchmark = "u4n":
                "F1" | "Jaccard"
            Default is `f1`
        benchmark : str, optional
            either "cs" for cs-benchmark (see [13])
            or "u4n" for Caicedos method (see [12])
            Default is `cs`
        threshold : float, optional
            Threshold for u4n. elem(0.5, 0.95)
            Default is `0.5`
    Retruns
    ----------
        out : ArrayLike
            Array of metric of len dts.
    '''
    if benchmark == 'cs':
        results = wrapper_cs(dts, gt)
        metric_val = results[metric]
        arr = np.append(arr, metric_val)

    if benchmark == 'u4n':
        for dt in dts:
            results, _, __ = wrapper_u4n(dt, gt)
            print(results)
            metric_val = results[np.round(results['Threshold'], 2) == threshold][metric]
            print(metric_val)
            arr = np.append(arr, metric_val)

    return arr


def cross_eval(
    results: Union[str, os.PathLike, PosixPath],
    # run: Union[str, os.PathLike, PosixPath],
    methods: list,
    section: str,
    metric: str = 'f1',
    benchmark: str = 'cs',
    threshold: int = 0.5,
    gt_path: Union[str, os.PathLike, PosixPath] = None,
    xenium: bool = False,
) -> tuple[ArrayLike, list]:
    '''Evaluate each mask against every other.

    Parameters
    ----------
        results : Path
            Path to directory containing all results.
        # run : Path
            # path to directory for run metrics and logs.
        methods : list
            List of all methods used for segmentation.
        section : str
            String of section evaluation is running on.
        metric : str, optional
            if benchmark = "cs":
                "f1" | "seg" | "jaccard" | "dice" | "PQ"
            if benchmark = "u4n":
                "F1" | "Jaccard"
            Default is `f1`
        benchmark : str, optional
            either "cs" for cs-benchmark (see [13])
            or "u4n" for Caicedos method (see [12])
            Default is `cs`
        threshold (default: 0.5): Threshold for u4n. elem(0.5, 0.95)

    Returns
    ----------
        out : tuple
            2D array of metric, labels ordered by evaluation
    '''
    arr = np.array([], dtype=float)

    gts = []
    labels = []

    processed = Path(f'{results}').parent / 'processed'
    file = Path(f'{processed}/{section}/morphology/focus/focus.ome.tif')
    img = tifffile.imread(file)
    shape = img.shape[:2]

    # if gt_path:
    #     gt = tifffile.imread(gt_path)
    #     labels.append('GT')

    # if xenium:
    #     output_path = Path(f'{results}/xenium/output/{section}')
    #     file = 'cell_polygons.geojson'
    #     wrap_ptm(output_path / file, output_path, shape)
    #     labels.append('xenium')

    methods = list(methods)

    for method in methods:
        # if method == 'proseg':
        #     files = [
        #         'cell-polygons.geojson.gz',
        #         'cell-polygons_layers.geojson.gz'
        #     ]
        #     for file in files:
        #         polygons_path = Path(
        #             f'{results}/{method}/output/{section}/{file}'
        #         )
        #         output_path = Path(
        #             f'{results}/{method}/output/{section}'
        #         )
        #         wrap_ptm(polygons_path, output_path, shape)
        # if method == 'segger':
        #     for mode in ['cell', 'nucleus']:
        #         file = f'boundaries_{mode}.geojson'
        #         output_path = Path(
        #             f'{results}/{method}/output/{section}/'
        #         )
        #         polygons_path = Path(
        #             output_path / f'{file}'
        #         )
        #         wrap_ptm(polygons_path, output_path, shape, mode=mode)
        files = list(
            Path(
                f'{results}/{method}/output/{section}/'
            ).glob('prediction*.tif')
        )
        files.sort()
        for path in files:
            if method == 'mesmer':
                gt = tifffile.imread(path)[0, ...]
            else:
                gt = tifffile.imread(path)

            gts.append(np.squeeze(gt))

            if method != 'dinocell':
                labels.append(method + path.stem[path.stem.rfind('_'):])
            else:
                labels.append(method)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        res = pool.map(functools.partial(
            eval_mask,
            dts=gts,
            arr=arr,
            metric=metric,
            benchmark=benchmark,
            threshold=threshold
        ), gts)
        pool.close()
        pool.join()
    res = np.vstack(res, dtype=float)
    # res_dict = dict(zip(labels, res))
    # with open(f'{results}/cross_evaluation.json') as file:
    #     json.dump(res_dict, file)
    return res, labels


# def plot_precision_recall(precision, recall, methods, sample_ids):
#     points = {m: [] for m in methods}

#     for sid in sample_ids:
#         for i, rad in enumerate(radii):
#             for m in methods:
#                 p = precision[sid][rad][m]
#                 r = recall[sid][rad][m]
#                 points[m].append((r, p))

#     fig, ax = plt.subplots(1, len(methods), figsize=(20, 2))
#     fig.text(0.5, -0.1, 'precision', ha='center', va='center')
#     fig.text(0.1, 0.5, 'recall', ha='center', va='center', rotation='vertical')
#     for method, a in zip(methods, fig.axes):
#         a.set_xlim(0.3, 1)
#         a.set_ylim(0.5, 1)
#         a.set_title(method)

#         xs = [x for x, y in points[method]]
#         ys = [y for x, y in points[method]]

#         sns.kdeplot(x=xs, y=ys, clip=(0, 1), ax=a)
#         a.scatter(xs, ys)
#     plt.show()
