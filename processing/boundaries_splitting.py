from pathlib import Path
import configparser
import argparse

import tomlkit
import json
import gzip

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

# types
from pandas.core.frame import DataFrame

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
    df: DataFrame,
    regions: dict
) -> DataFrame:
    """Assign region to coordniates in DataFrame.
    Args:
        df: DataFrame with x and y vertex.
        regions: Dictionary with coordinates of bbox.
    Returns:
        DataFrame with each row assigned to a region in 'regions'.
    """
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

    return df


def relative(
    df: DataFrame,
    region_data: dict
) -> DataFrame:
    """Subtract region origin from vertex.
    Args:
        df: DataFrame with x and y vertex.
        regions_data: Dictionary with coordinates of bbox-corners.
    Returns:
        DataFrame with coordinates relative to region origin.
    """
    y_loc = df.columns.get_loc('vertex_y')
    x_loc = df.columns.get_loc('vertex_x')

    vertex_y_arr = df['vertex_y'].to_numpy()
    vertex_y_arr = np.nan_to_num(vertex_y_arr, nan=0, posinf=0, neginf=0)
    vertex_y_arr_r = vertex_y_arr - region_data['y_min']
    vertex_y_arr_r.astype(np.int64)
    vertex_y_arr_r[vertex_y_arr_r == 0] = np.nan

    df.iloc[:,y_loc] = pd.DataFrame(vertex_y_arr_r)

    vertex_x_arr = df['vertex_x'].to_numpy()
    vertex_x_arr = np.nan_to_num(vertex_x_arr, nan=0, posinf=0, neginf=0)
    vertex_x_arr_r = vertex_x_arr - region_data['x_min']
    vertex_x_arr_r.astype(np.int64)
    vertex_x_arr_r[vertex_x_arr_r == 0] = np.nan

    df.iloc[:,x_loc] = pd.DataFrame(vertex_x_arr_r)

    return df


def pixelate(
    df: DataFrame,
    pixelsize: float
) -> DataFrame:
    """Devide by pixelsize.
    Args:
        df: DataFrame with x and y vertex.
        pixelsize: Pixelsize of XY [unit of image]/px.
    Returns:
        DataFrame with coordinates in pixel coordinates.
    """
    y_loc = df.columns.get_loc('vertex_y')
    x_loc = df.columns.get_loc('vertex_x')

    vertex_y_arr = df['vertex_y'].to_numpy()
    vertex_y_arr = np.nan_to_num(vertex_y_arr, nan=0, posinf=0, neginf=0)
    vertex_y_arr_p = vertex_y_arr / pixelsize
    vertex_y_arr_p.astype(np.int64)
    vertex_y_arr_p[vertex_y_arr_p == 0] = np.nan

    df.iloc[:,y_loc] = pd.DataFrame(vertex_y_arr_p)
    
    vertex_x_arr = df['vertex_y'].to_numpy()
    vertex_x_arr = np.nan_to_num(vertex_x_arr, nan=0, posinf=0, neginf=0)
    vertex_x_arr_p = vertex_x_arr / pixelsize
    vertex_x_arr_p.astype(np.int64)
    vertex_x_arr_p[vertex_x_arr_p == 0] = np.nan

    df.iloc[:,x_loc] = pd.DataFrame(vertex_x_arr_p)

    return df


def save_section(
    region_name: str,
    region_data: dict,
    df: DataFrame,
    bound: str='cell'
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
    # main selection
    sub_results_df = df[df['region'] == region_name]

    # remove region offset
    sub_results_df = relative(sub_results_df, region_data)

    # pixelation
    sub_results_df = pixelate(sub_results_df)

    # cleanup
    # columns = [
    #     c for c in sub_results_df.columns if c != 'region'
    # ]
    columns = list(sub_results.columns).remove('region')
    sub_results_df = sub_results_df.loc[:,columns]

    if sub_results_df.size == 0:
        print(f'region {region_name}: no datapoints matching')
    else:
        # save thingy
        sub_results_pq = pa.Table.from_pandas(sub_results_df)
        del sub_results_df

        output_dir = processed / f'{region_name}/boundaries/'
        output_dir.mkdir(parents=True, exist_ok=True)

        if bound == 'cell':
            f_str = 'cell'
        else:
            f_str = 'nucleus'
        
        parquet_path = output_dir / '{0}_relative.parquet'.format(f_str)
        pq.write_table(sub_results_pq, parquet_path)

        print(f'region {region_name}: saved results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    paths = config['paths']

    home = paths['home']
    data = Path(paths['data_path'])
    sample = paths['name']
    ## define processed directory    
    processed = Path(f'{home}{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = paths['sections_path']
    else:
        sections_path = processed / 'sections_px.json'

    # define variables
    pixelsizeXY = imagestats['pixelsize_xy']

    # load sections_dictionary
    with open(sections_path) as f:
        section_dictionary = json.load(f)

    regions = define_regions_to_extract(section_dictionary, pixelsizeXY) 

    for file in Path(data).glob('*_boundaries.parquet'):
        parquet_file = pq.ParquetFile(file)
        bound = file.strip('_boundaries.parquet')
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            with parquet_file.iter_batches() as reader:
                results = pool.imap(
                    functools.partial(
                        process_chunk,
                        regions=regions
                    ), reader
                )
                pool.close()
                pool.jon()
                results_df = pd.concat(results)

        with mp.Pool(processes=mp.cpu_count()-1) as pool:
        pool.imap_unordered(
            functools.partial(
                save_section,
                df=results_df,
                regions=regions,
                bound=bound
            ),
            regions.keys()
        )
        pool.close()
        pool.join()