"""
Section the transcripts.
Transcripts are saved as csv.gz | relative micrometer coordinates
# Boundaries as qarquets | relative pixel coordinates
"""

from pathlib import Path
import configparser
import functools
import argparse
import tomllib

import json
import gzip

# from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
import pandas as pd
import numpy as np


def regions_to_extract(sections_dict, pixelsizeXY):
    """Change unit of regions of interest from dictionary.
    Args:
        sections_dict: dictionary containing the coordinates, in px,
                       of the regions of interest in the smaple.
        pixelsize: tuple of size of one pixel in x,y dimension.
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

def relative(df, region_data):
    """Removes y/x min -> relative coordinates.

    Args:
        df: DataFrame
        region_data: coordinates of region working on.
    Returns:
        DataFrame with new y/x values.
    """
    df['y_location'] = (df['y_location'] - region_data['y_min'])
    df['x_location'] = (df['x_location'] - region_data['x_min'])
   
    return df

def pixelate(df, pixelsize):
    """
    Original values are in micrometers. Changes these to be in pixels.

    Args:
        df: The dataframe to pixelate.
        pixelsize: Tuple of pixelsizes in x/y and z dimension.
    Returns:
        A pixelated dataframe in x,y and z.
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

def process_chunk(df, regions):
    """Assigns the y/x values to the respective bin. Adds a MultiIndex to the df
    Args:
        df: The dataframe to bin.
        regions: regions of interest and coordinates as 
                 dictionary of dictionaries.
    Returns:
        Dataframe with additional column of region names.
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


def save_section(df, region_name, regions):
    """Process the DataFrame and save it as .csv and gzip compressed.
    Args:
        df: dataframe to process and save.
        region_name: name of region. str.
        regions: Dictionary of Dictionaries with 
                 coordinates in micrometers
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

        sub_results_df.to_csv(csv_path / 'relative.csv', index=False)

        # compress for ProSeg
        sub_results_df.to_csv(output_dir / 'relative.csv.gz', index=False,
                              compression='infer'
        )

        print(f'region {region_name}: saved restults')

if __name__ == '__main__':

    # define paths
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']

    home = paths['home']

    # directory for saving!
    processed = Path(f'{home}/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    # define variables
    pixelsizeXY = imagestats['pixelsize_xy']
    pixelsizeZ = imagestats['pixelsize_z']

    with open(processed / 'sections_px.json', 'r') as f:
        sections_dict = json.load(f)

    dtype_dict = dict(
        zip(['transcript_id','overlaps_nucleus','codeword_index'],
            [np.int64]*3
        )
    )

    regions = regions_to_extract(sections_dict, pixelsizeXY)
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
