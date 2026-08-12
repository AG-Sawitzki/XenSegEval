from XenSegEval.utils import (
    get_config_args,
    get_section_coords
)
from XenSegEval.processing.utils import (
    wrap_table_actions
)

import multiprocessing as mp
from pathlib import Path
import argparse

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
    parser.add_argument(
        '-m', '--Mode',
        choices=['cell', 'nucleus'],
        help='The segmentation mode of segger.'
    )

    args = parser.parse_args()

    config_path = args.Config
    mode = args.Mode

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'segger')
    globals().update(variables)

    path = Path(
        f'{results}/{mode}/cell_boundaries.parquet'
    )
    for section in sections:
        section_dict = section_dictionary[section]
        sub_gdf = wrap_table_actions(
            path, action='location', 
            section_dict=section_dict, 
            pixelsize_xy=pixelsizeXY
        )
        print(sub_gdf)
        out = Path(f'{results}/{section}')
        out.mkdir(parents=True, exist_ok=True)
        sub_geoj = sub_gdf.to_json()

        sub_gdf_relative = wrap_table_actions(
            sub_gdf, action='relative',
            section_dict=section_dict
        )
        print('Relative')
        sub_geoj_relative = sub_gdf_relative.to_json()
        with open(out/f'{mode}_polygons.geojson', 'w') as f:
            f.write(sub_geoj_relative)