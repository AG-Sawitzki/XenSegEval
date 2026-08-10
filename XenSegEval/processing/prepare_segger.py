from XenSegEval.utils import (
    get_config_args,
    get_section_coords
)
from XenSegEval.processing import (
    wrap_parquet_actions
)

import multiprocessing as mp
from pathlib import Path
import configparser
import functools
import argparse
import sys
import os

import tomlkit
import json
import gzip

import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl
import pandas as pd
import numpy as np
import shapely

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prepare_segger')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'segger')
    globals().update(variables)

    for section in sections:
        for mode in method['prediction']['prediction-mode']:
            table = pq.read_table(
                f'{results}/{mode}/cell_boundaries.parquet'
            )
            gdf = wrap_parquet_actions(table, filter_type='wkb')

            if 'cell_id' in gdf.columns:
                gdf.drop('cell_id', axis=1)

            x_coords, y_coords = get_section_coords(section_dictionary, section)
            x_min, x_max = x_coords
            y_min, y_max = y_coords
            polygon = shapely.Polygon([
                (x_min, y_min), (x_max, y_min),
                (x_min, y_max), (x_max, y_max),
            ])
            check = gdf['geometry'].within(polygon)
            sub_gdf = gdf[check]
            sub_gdf.to_file(
                f'{results}/segger/output/{section}/boundaries_{mode}.geojson',
                driver='GEOJson'
            )