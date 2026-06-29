# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)
# for cs-bench
from XenSegEval.eval.cs_benchmark.metrics import Metrics
# plotting
from XenSegEval.plotting.utils import heatmap, annotate_heatmap
# Utils
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    prepare_ProSeg,
    polygon_to_mask,
    cross_eval
)

import sys
import gzip
import pickle
import argparse
from pathlib import Path

import tomlkit
import numpy as np
import pandas as pd
import tifffile as tf
import geopandas as gpd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CrossEval.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'cross')
    globals().update(variables)

    run = Path(f'{home}/{sample_name}/run/')

    for section in sections:
        if CROSS['use']:
            res, labels = cross_eval(
                results,
                run,
                methods,
                section,
                metric=CROSS['metric'],
                benchmark=CROSS['benchmark'],
                threshold=CROSS['threshold']
            )

        print(res)

        np.save(f'{results}/cross_evaluation.npy', res)

        fig, ax = plt.subplots()

        im, cbar = heatmap(
            res, labels, labels, ax=ax,
            cmap='YlOrRd', cbarlabel=CROSS['metric']
        )
        texts = annotate_heatmap(im, valfmt='{x:.1f}')
        fig.tight_layout()
        fig.savefig(
            f'/data/cephfs-1/home/users/juno12_c/'
            f'cross_{CROSS['benchmark']}_{CROSS['metric']}.png',
            dpi=250
        )
