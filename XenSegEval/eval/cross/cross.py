# for jaccard
# from XenSegEval.eval.unet4nuclei.evaluation import (
#     compute_af1_results,
#     get_false_negatives,
#     get_splits_and_merges
# )
# for cs-bench
# from XenSegEval.eval.cs_benchmark.metrics import Metrics
# plotting
# from XenSegEval.plotting.utils import heatmap, annotate_heatmap
# Utils
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    # prepare_ProSeg,
    # polygon_to_mask,
    cross_eval,
    mean_cross_eval
)

# import sys
# import gzip
# import pickle
import argparse
from pathlib import Path

import tomlkit
import numpy as np
# import pandas as pd
# import tifffile as tf
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CrossEval.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    # parser.add_argument(
    #     '-p', '--Plot',
    #     action='store_true',
    #     help='Use if cross evaluation should be plotted after calculation.'
    # )
    args = parser.parse_args()

    config_path = args.Config
    # plot = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'cross')
    globals().update(variables)

    run = Path(f'{home}/{sample_name}/run/')

    if CROSS['use']:
        for section in sections:
            metric = CROSS['metric']
            if str(metric).istitle():
                benchmark = 'u4n'
            else:
                benchmark = 'cs'
            res, labels = cross_eval(
                results,
                # run,
                methods,
                section,
                metric=metric,
                benchmark=benchmark,
                threshold=CROSS['threshold']
            )

            print(res)

            cross_with_avg, avg_hori, avg_vert = mean_cross_eval(
                res,
                methods,
                labels
            )

            print(cross_with_avg)

            path = (
                f'{results}/{CROSS["metric"]}_cross_evaluation_{section}'
            )
            np.save(path + '.npy', res)
            np.save(path + '_avg.npy', cross_with_avg)
            with open(path, 'w') as f: 
                f.write(' '.join(labels))
            # if plot:
            #     fig, ax = plt.subplots()

            #     im, cbar = heatmap(
            #         res, labels, labels, ax=ax,
            #         cmap='YlOrRd', cbarlabel=CROSS['metric']
            #     )
            #     texts = annotate_heatmap(im, valfmt='{x:.1f}')
            #     fig.tight_layout()
            #     fig.savefig(
            #         f'/data/cephfs-1/home/users/juno12_c/'
            #         f'cross_{CROSS['benchmark']}_{CROSS['metric']}_{section}.pdf'
            #     )
