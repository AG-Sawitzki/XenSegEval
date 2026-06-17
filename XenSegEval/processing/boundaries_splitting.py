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
import pandas as pd
import numpy as np

# types
from typing import Any
from pandas.core.frame import DataFrame

from XenSegEval.utils import get_config_args


def define_regions_to_extract(
    sections_dict: dict,
    pixelsizeXY: float
) -> dict:
    """Restructure the sectios_dictionary.
    Args:
        sections_dict: Dictionary of bounding boxes.
        pixelsizeXY: Float of size of one pixel in x,y dimension.
    Returns:
        Reorganized and refactored coordinates of bbox as dictionary.
    """
    regions = {}
    
    for region, bbox in sections_dict.items():
        y_min_px, x_min_px = bbox[0]
        y_max_px, x_max_px = bbox[1]

        regions[region] = {
            'y_min': y_min_px * pixelsizeXY,
            'x_min': x_min_px * pixelsizeXY,
            'y_max': y_max_px * pixelsizeXY,
            'x_max': x_max_px * pixelsizeXY,
        }

    return regions


def process_chunk(
    df: Any,
    regions: dict
) -> DataFrame:
    """Assign region to coordniates in DataFrame.
    Args:
        df: Parquet with x and y vertex.
        regions: Dictionary with coordinates of bbox.
    Returns:
        DataFrame with each row assigned to a region in 'regions'.
    """
    df = df.to_pandas()
    regions_mapping = pd.Series(index=df.index, dtype=str).fillna('')

    for region_name, region_data in regions.items():
        y_min = region_data['y_min']
        x_min = region_data['x_min']
        y_max = region_data['y_max']
        x_max = region_data['x_max']

        regions_mapping[
            (x_min <= df['vertex_x'])
            & (df['vertex_x'] <= x_max)
            & (y_min <= df['vertex_y'])
            & (df['vertex_y'] <= y_max)
        ] = region_name

    df['region'] = regions_mapping
    # print(df.head(n=5))
    return df


def relative(
    df: Any,
    region_data: Any
) -> DataFrame:
    """Subtract region origin from vertex.
    Args:
        df: DataFrame with x and y vertex.
        regions_data: Dictionary with coordinates of bbox-corners.
    Returns:
        DataFrame with coordinates relative to region origin.
    """
    df['vertex_y'] = (df['vertex_y'] - region_data['y_min'])
    df['vertex_x'] = (df['vertex_x'] - region_data['x_min'])
    return df


def pixelate(
    df: Any,
    pixelsize: Any
) -> DataFrame:
    """Devide by pixelsize.
    Args:
        df: DataFrame with x and y vertex.
        pixelsize: Pixelsize of XY [unit of image]/px.
    Returns:
        DataFrame with coordinates in pixel coordinates.
    """
    df['vertex_y'] = (
        df['vertex_y'] / pixelsizeXY
    ).round(0).astype(np.int64)
    
    df['vertex_x'] = (
        df['vertex_y'] / pixelsizeXY
    ).round(0).astype(np.int64)   
    # print(df.head(n=5))
    return df


def save_section(
    region_name: Any,
    regions: Any,
    df: Any,
    pixelsizeXY: Any,
    bound: Any = 'cell'
) -> None:
    """Saves the DataFrame as parquet.
    Args:
        region_name: Key of regions for region to save.
        region_data: Dictionary with coordinates of bbox-corners.
        df: DataFrame to save a region of.
        bound: Either 'cell' or 'nucleus'.
               Defines which boundaries file was read
               and adds an identifier to the path.
    Returns:
        None.   
    """
    region_data = regions[region_name]
    sub_results_df = df[df['region'] == region_name]

    # remove region offset
    sub_results_df = relative(sub_results_df, region_data)

    # pixelation
    sub_results_df = pixelate(sub_results_df, pixelsizeXY)

    sub_results_df.drop(columns='region', inplace=True)
    if sub_results_df.size == 0:
        print(f'region {region_name}: no datapoints matching')
    else:
        # save thingy
        sub_results_pq = pa.Table.from_pandas(sub_results_df, preserve_index=False)
        # print(sub_results_pq)
        # del sub_results_df

        output_dir = Path(processed / f'{region_name}/boundaries/')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(output_dir)
        f_str = str(bound)

        parquet_path = Path(output_dir / f'{f_str}_relative.parquet')
        pq.write_table(sub_results_pq, str(parquet_path))

        print(f'region {region_name}: saved results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='boundaries')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'boundaries')
    globals().update(variables)

    regions = define_regions_to_extract(section_dictionary, pixelsizeXY) 

    for file in Path(data_path).glob('*_boundaries.parquet'):
        parquet_file = pq.ParquetFile(file)
        bound = str(file).removesuffix('_boundaries.parquet')
        bound = str(bound).removeprefix(str(data_path)+'/')
        print(bound)
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            # with parquet_file.iter_batches() as reader:
            results = pool.imap(
                functools.partial(
                    process_chunk,
                    regions=regions
                ), parquet_file.iter_batches() # reader
            )
            pool.close()
            pool.join()
            results_df = pd.concat(results)
        print(results_df.head(n=5))
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            pool.imap(
                functools.partial(
                    save_section,
                    df=results_df,
                    regions=regions,
                    pixelsizeXY=pixelsizeXY,
                    bound=bound
                ),
                regions.keys()
            )
            pool.close()
            pool.join()