# Utils
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    wrapper_cs,
    wrapper_u4n
)

import sys
import gzip
from pathlib import Path
import argparse
import pickle

import tifffile as tf
import pandas as pd
import numpy as np
import tomlkit
import geopandas as gpd


# for pca
# from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.morphology import label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Eval.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    parser.add_argument(
        '-gts', '--GTSection',
        help='The section name corresponding to the Ground Truth.'
    )
    args = parser.parse_args()

    method = args.Method
    config_path = args.Config
    section = args.GTSection

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    focus_path = Path(f'{processed}/{section}/morphology/focus/')

    gt = tf.imread(gt_path)

    mask_path = f'{results}/{method}/output/{section}/'

    for file in Path(mask_path).glob('prediction*.tif'):
        print(file)
        mask = tf.imread(file)

        stem = file.stem
        if '_' in stem:
            dir_name = stem[stem.rfind('_'):]
            outdir = Path(
                f'{results}/{method}/'
                f'evaluation/{section}/{dir_name}'
            )
        else:
            outdir = Path(
                f'{results}/{method}/evaluation/{section}'
            )

        outdir.mkdir(parents=True, exist_ok=True)

        if method == 'mesmer':
            mask = mask[0, ...]
            mask = np.squeeze(mask)
            print(mask.shape, gt.shape)

        if JACCARD['use']:
            print('jaccard')
            results, false_negatives, split_merges = wrapper_u4n(
                mask,
                gt,
                method=method
            )
            results.to_csv(outdir / 'results.csv', index=False)
            false_negatives.to_csv(outdir / 'false_negatives.csv', index=False)
            split_merges.to_csv(outdir / 'split_merges.csv', index=False)

        if CS_BENCH['use']:
            print('cs_bench')
            results = wrapper_cs(mask, gt, method=method)
            results.to_csv(outdir / 'CS-BENCH.csv', index=False)
            print('saved')
