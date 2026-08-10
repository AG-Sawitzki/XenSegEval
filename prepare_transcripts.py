from XenSegEval.utils import get_config_args
from XenSegEval.processing.utils import (
    wrap_filter
)

import multiprocessing as mp
from pathlib import Path
import functools
import argparse
import sys
import os

import tomlkit
import json
import gzip

# from numpy.lib.stride_tricks import sliding_window_view
import polars as pl
import pandas as pd
import numpy as np

# types
from typing import Any, Union
from pandas.core.frame import DataFrame



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='transcripts')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'transcripts')
    globals().update(variables)

    df = pl.read_parquet(f'{data_path}/transcripts.parquet')

    print('Processing Chunks.')
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.imap(
            functools.partial(
                wrap_filter,
                table=df,
                pixelsize_xy,
                processed,
            ),
            section=sections_dictionary.items()
        )
        pool.close()
        pool.join()
