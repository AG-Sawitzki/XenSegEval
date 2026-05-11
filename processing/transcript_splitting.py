"""
Section the transcripts.
Transcripts are saved as csv.gz | relative micrometer coordinates
# Boundaries as qarquets | relative pixel coordinates
"""

from pathlib import Path
import configparser
import functools
import argparse

import json
import gzip

# from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
import pandas as pd
import numpy as np

# def find_bins(bins, section):
#     """
#     Find the bins the section fits into.

#     Args:
#         bins:    the bins...
#         section:    string of the section number.

#     Returns:
#         Strings for the bins.
#     """
#     y_bins, x_bins = bins

#     y_bins_v = sliding_window_view(y_bins, (2,))
#     x_bins_v = sliding_window_view(x_bins, (2,))

#     coord = np.array(sections_dict[section])*pixelsize

#         # check in which bin
#     y_bin_index = np.where(y_bins_v <= coord[:,0])[0][-1]
#     x_bin_index = np.where(x_bins_v <= coord[:,1])[0][-1]

#     return 'y{}'.format(y_bin_index), 'x{}'.format(x_bin_index)

# def make_bins(sections_dict):
#     """
#     Makes the bins based on the max and min values in sections_dict.
#     It expects 3 samples along x and 4 along y.

#     Args:
#         sections_dict: the dictionary containing the upper left and lower right corner of the samples

#     Returns:
#         List of two arrays for the bins. [y_bin, x_bin]
#     """
#     keys = sections_dict.keys()
#     values = np.array([sections_dict[i] for i in keys])*pixelsize

#     y_min, x_min = np.min(values, axis=0)[0]
#     y_max, x_max = np.max(values, axis=0)[1]

#     y_bins = np.linspace(y_min, y_max, num = 5)
#     x_bins = np.linspace(x_min, x_max, num = 4)

#     for i in range(len(keys)):
#         y_check = sliding_window_view(y_bins, (2,)) >= values[i,:,0]
#         x_check = sliding_window_view(x_bins, (2,)) >= values[i,:,1]
#         if y_check.any() and x_check.any():
#             continue
#         else:
#             print(f'Bin too small for section {i}')

#     return [y_bins, x_bins]

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
    # df.loc[:,'x_location':'y_location'] = (df.loc[:,'x_location':'y_location'].to_numpy() - sections_dict[section][0]).round(0).astype(np.int64)

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

    # arr = (df.loc[:,'x_location':'y_location'].to_numpy() / pixelsize).round(0).astype(np.int64)
	# arr.round(0)
	# df.loc[:,'x_location':'y_location'] = arr.astype(np.int64)

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

    regions_mapping = pd.Series(index=df.index, dtype=str). fillna('')

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

    df['region'] = regions_mapping

    # y_binned = pd.cut(df['y_location'], bins[0], labels = ['y0', 'y1', 'y2', 'y3'], include_lowest = True).to_numpy()
    # x_binned = pd.cut(df['x_location'], bins[1], labels = ['x0', 'x1', 'x2'], include_lowest = True).to_numpy()

    # index_ = df.index.to_numpy()
    # index = pd.MultiIndex.from_arrays([y_binned, x_binned, index_], names = ('y_bin', 'x_bin', 'idx'))

    # df.index = index

    # df.reset_index(inplace = True)
    # df.dropna(axis = 0, inplace = True)
    # df.set_index(['y_bin', 'x_bin', 'idx'], inplace = True)
    # df.sort_index(inplace = True)

    # if df.index.is_monotonic_increasing != True:
    #     # print('sorted')
    #     # else:
    #     print('sorting already prob here')

    return df

# def process_SubFrame(df, section, sections_dict):
#     """
#     Processes the SubFrame (metabin extract of the DataFrame).
#         - Pixelates it.
#         - Assigns new bins based on upper left and lower right corner

#     Args:
#         df: a DataFrame
#         section: string of the section
#         sections_dict: dictionary of section coordinates

#     Returns:
#         DataFrame with values between y/x min/max.
#     """
#     print(df.loc[:10,'x_location':'y_location'])
#     df = pixelate(df)
#     print(df.loc[:10,'x_location':'y_location'])
#     print('px')

#     y_min, x_min = sections_dict[section][0]
#     y_max, x_max = sections_dict[section][1]
#     print('bins2')

#     y_binned = pd.cut(df['y_location'], [y_min,y_max+1], labels = ['y']).to_numpy()
#     x_binned = pd.cut(df['x_location'], [x_min,x_max+1], labels = ['x']).to_numpy()
#     print('binned2')
#     print(y_binned)
#     print(x_binned)

#     index = pd.MultiIndex.from_arrays([y_binned, x_binned], names = ('y_bin', 'x_bin'))
#     df.index = index
#     df.sort_index(inplace=True)
#     print('indexed')
#     print(df.head(n=5))

#     df = df.loc[('y','x')]
#     print('masked')

#     return df

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

    # y_bin, x_bin = find_bins(bins, section)
    # print('bins', y_bin, x_bin)
    # print(sections_dict[section])
    # sub_results_df = results_df.loc[(y_bin,x_bin)]
    # print(sub_results_df.head(n=10))
    # sub_results_df.reset_index(drop=True, inplace=True)
    # print('binned')

    sub_results_df = df[df['regions'] == region_name]
    print(sub_results_df.loc[:10,'x_location':'y_location'])

    # px_filtered_sub_results_df = process_SubFrame(sub_results_df, section, sections_dict)
    # print('masked and px')
    # relative_px_filtered_sub_results_df = relative(px_filtered_sub_results_df, sections_dict)
    # print('relative')

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

    config = configparser.ConfigParser()
    config.read(config_path)

    data = Path(config['PATHS']['data_path'])
    sample = config['PATHS']['sample_name']

    # directory for saving!
    processed = Path(f'/data/cephfs-2/unmirrored/groups/sawitzki/Juno/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    # define variables
    chunks = config['PREPROCESSING'].getfloat('chunks')
    min_size = config['PREPROCESSING'].getfloat('min_size')
    n_roi = config['PREPROCESSING'].getfloat('n_roi')
    overlap = config['PREPROCESSING'].getfloat('overlap')

    pixelsizeXY = config['ImageStats'].getfloat('pixelsizeXY')
    pixelsizeZ = config['ImageStats'].getfloat('pixelsizeZ')

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
