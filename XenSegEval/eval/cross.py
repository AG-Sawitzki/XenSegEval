# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)
# for cs-bench
from XenSegEval.eval.cs_benchmark.metrics import Metrics
# Utils
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    prepare_ProSeg,
    polygon_to_mask,
    cross_eval
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
    parser = argparse.ArgumentParser(prog='CrossEval.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'cross')
    globals().update(variables)

    section = 'newmem'

    print(results)

    run = Path(f'{home}/{sample_name}/run/')

    if CROSS:
        cross_eval(
            results,
            run,
            methods,
            section,
            threshold=0.5
        )
