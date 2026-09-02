from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    base_stats
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Eval.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    parser.add_argument(
        '-s', '--Section',
        help='The section name corresponding to the segmented area.'
    )
    args = parser.parse_args()

    method = args.Method
    config_path = args.Config
    section = args.Section

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    mask_path = f'{results}/{method}/output/{section}/'
    for file in Path(mask_path).glob('prediction*.npy'):
        mask = np.load(file, allow_pickle=True)

        stem = str(file.stem)
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

        res = base_stats(mask)
        res.write_csv(outdir / 'count_area.csv')

        print('saved')