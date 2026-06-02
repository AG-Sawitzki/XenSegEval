"""
Section the transcripts.
Transcripts are saved as csv.gz | relative micrometer coordinates
# Boundaries as qarquets | relative pixel coordinates
"""

from pathlib import Path
import configparser
import functools
import argparse

import tomlkit
import json
import gzip

# from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
import pyarrow as pa
import pandas as pd
import numpy as np


def define_regions_to_extract(
    sections_dict: dict,
    pixelsizeXY: float
) -> dict:
    """Change unit of regions of interest from dictionary.
    Args:
        sections_dict: Dictionary of bounding boxes.
        pixelsizeXY: Float of size of one pixel in x,y dimension.
    Returns:
        Dictionary of Dictionaries with coordinates in micrometers.
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
    region_mapping = pd.Series(index=df.index, dtype=str). fillna('')

    for region_name, region_data in regions.items():
        y_min = region_data['y_min']
        x_min = region_data['x_min']
        y_max = region_data['y_max']
        x_max = region_data['x_max']

        region_mapping[
            (x_min <= df['x_location'])
            & (df['x_location'] <= x_max)
            & (y_min <= df['y_location'])
            & (df['y_location'] <= y_max)
        ] = region_name

    df['region'] = region_mapping

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

    y_arr = df['y_location'].to_numpy()
    y_arr = np.nan_to_num(y_arr, nan=0, posinf=0, neginf=0)
    y_arr_r = y_arr - region_data['y_min']
    y_arr_r.astype(np.int64)
    y_arr_r[y_arr_r == 0] = np.nan

    df.iloc[:,y_loc] = pd.DataFrame(y_arr_r)

    x_arr = df['x_location'].to_numpy()
    x_arr = np.nan_to_num(x_arr, nan=0, posinf=0, neginf=0)
    x_arr_r = x_arr - region_data['x_min']
    x_arr_r.astype(np.int64)
    x_arr_r[x_arr_r == 0] = np.nan

    df.iloc[:,x_loc] = pd.DataFrame(x_arr_r)

    #df['y_location'] = (df['y_location'] - region_data['y_min'])
    #df['x_location'] = (df['x_location'] - region_data['x_min'])

    return df

def pixelate(
    df: DataFrame,
    pixelsize: tuple
) -> DataFrame:
    """Devide by pixelsize.
    Args:
        df: DataFrame with x and y vertex.
        pixelsize: Pixelsize of XY and Z [unit of image]/px.
    Returns:
        DataFrame with coordinates in pixel coordinates.
    """ 
    df['y_location'] = (
        df['y_location'] / pixelsize[0]
    ).round(0).astype(np.int64)

    df['x_location'] = (
        df['x_location'] / pixelsize[0]
    ).round(0).astype(np.int64)

    df['z_location'] = (
        df['z_location'] / pixelsize[1]
    ).round(0).astype(np.int64)

    return df


def save_section(
    region_name: str,
    region_data: dict,
    df: DataFrame,
) -> None:
    """Saves the DataFrame as .csv, gzip compressed and parquet.
    Args:
        region_name: Key of regions for region to save.
        region_data: Dictionary with coordinates of bbox-corners.
        df: DataFrame to save a region of.
    Returns:
        None.
    """
    region_data = regions[region_name]

    sub_results_df = df[df['regions'] == region_name]
    print(sub_results_df.loc[:10,'x_location':'y_location'])

    sub_results_df = relative(sub_results_df, region_data)

    sub_results_df = pixelate(
        sub_results_df,
        pixelsize=(pixelsizeXY, pixelsizeZ)
    )

    sub_results_df.drop(columns='regions', inplace=True)

    if sub_results_df.size == 0:
        print(f"region {region_name}: no datapoints matching")
    else:
        # save thingy
        output_dir = Path(processed / '{0}/transcripts/'.format(region_name))
        output_dir.mkdir(parents=True, exist_ok=True)

        sub_results_df.to_csv(output_dir / 'relative.csv', index=False)

        # compress for ProSeg
        sub_results_df.to_csv(
            output_dir / 'relative.csv.gz', index=False,
            compression='infer'
        )
        sub_results_pq = pa.Table.from_pandas(sub_results_df)
        pa.parquet.write_table(
            sub_results_pq, output_dir / 'relative.parquet'
        )

        print(f'region {region_name}: saved restults')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    paths = config['paths']
    imagestats = config['ImageStats']

    home = paths['home']
    data = Path(paths['data_path'])
    sample = paths['sample_name']
    ## define processed directory 
    processed = Path(f'{home}/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = paths['sections_path']
    else:
        sections_path = processed / 'sections_px.json'

    # define variables
    pixelsizeXY = imagestats['pixelsize_xy']
    pixelsizeZ = imagestats['pixelsize_z']

    # load sections_dictionary
    with open(sections_path) as f:
        section_dictionary = json.load(f)

    dtype_dict = dict(
        zip(['transcript_id','overlaps_nucleus','codeword_index'],
            [np.int64]*3
        )
    )

    regions = regions_to_extract(section_dictionary, pixelsizeXY)
    print(regions)

    print('Processing Chunks.')
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        with pd.read_csv(
            data / 'transcripts.csv.gz',
            compression = 'infer',
            dtype=dtype_dict,
            chunksize = 20000
        ) as reader:
            results = pool.imap(
                functools.partial(
                    process_chunk,
                    regions=regions
                ),
                reader
            ) # chunksize = int(20000/mp.cpu_count()-1))
            pool.close()
            pool.join()
            results_df = pd.concat(results)
    # results_df.index = results_df.index.sort_values()
    print(results_df.head(n=5))

    # save the sections
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        pool.imap_unordered(
            functools.partial(
                save_section,
                df=results_df,
                regions=regions
            ),
            regions.keys()
        )
        pool.close()
        pool.join()
