from pathlib import Path
import configparser
import argparse

import json
import gzip

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np


def define_regions_to_extract(sections_dict):

    regions = {}
    
    for region, bbox in sections_dict.items():
        y_min_px, x_min_px = bbox[0]
        y_max_px, x_max_px = bbox[1]

        regions[region] = {
            'y_min': y_min_px * pixelsize,
            'x_min': x_min_px * pixelsize,
            'y_max': y_max_px * pixelsize,
            'x_max': x_max_px * pixelsize,
        }

    return regions


def process_chunk(df, regions):

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

    # print(list(set(regions_mapping)))

    df['region'] = regions_mapping

    return df


def relative(df, region_data):
    vertex_y_arr = df['vertex_y'].to_numpy()
    vertex_y_arr = np.nan_to_num(vertex_y_arr, nan=0, posinf=0, neginf=0)
    vertex_y_arr_r = vertex_y_arr - region_data['y_min']
    vertex_y_arr_r.astype(np.int64)
    vertex_y_arr_r[vertex_y_arr_r == 0] = np.nan

    df.iloc[:,2] = pd.DataFrame(vertex_y_arr_r)

    vertex_x_arr = df['vertex_x'].to_numpy()
    vertex_x_arr = np.nan_to_num(vertex_x_arr, nan=0, posinf=0, neginf=0)
    vertex_x_arr_r = vertex_x_arr - region_data['x_min']
    vertex_x_arr_r.astype(np.int64)
    vertex_x_arr_r[vertex_x_arr_r == 0] = np.nan

    df.iloc[:,1] = pd.DataFrame(vertex_x_arr_r)
    
    # df.iloc[:, 1:3] = pd.DataFrame((
    #     df.iloc[:, 1:3].to_numpy()
    #     - [region_data['x_min'],region_data['y_min']]
    # ))
    
    return df


def pixelate(df):    
    vertex_y_arr = df['vertex_y'].to_numpy()
    vertex_y_arr = np.nan_to_num(vertex_y_arr, nan=0, posinf=0, neginf=0)
    vertex_y_arr_p = vertex_y_arr / pixelsize
    vertex_y_arr_p.astype(np.int64)
    vertex_y_arr_p[vertex_y_arr_p == 0] = np.nan
    df.iloc[:,2] = pd.DataFrame(vertex_y_arr_p)
    
    vertex_x_arr = df['vertex_y'].to_numpy()
    vertex_x_arr = np.nan_to_num(vertex_x_arr, nan=0, posinf=0, neginf=0)
    vertex_x_arr_p = vertex_x_arr / pixelsize
    vertex_x_arr_p.astype(np.int64)
    vertex_x_arr_p[vertex_x_arr_p == 0] = np.nan
    df.iloc[:,1] = pd.DataFrame(vertex_x_arr_p)
    # df.iloc[:, 1:3] = pd.DataFrame((
    #     df.iloc[:, 1:3].to_numpy()
    #     / pixelsize
    # ).round(0), dtype='Int64')
    
    return df


def save_section(region_name, region_data, df):
    # main selection
    sub_results_df = df[df['region'] == region_name]

    # remove region offset
    sub_results_df = relative(sub_results_df, region_data)

    # pixelation
    sub_results_df = pixelate(sub_results_df)

    # cleanup
    columns = [
        c for c in sub_results_df.columns if c != 'region'
    ]
    # alternative: columns = list(sub_results.columns).remove('region')
    sub_results_df = sub_results_df.loc[:,columns]
    # subresults_df.drop(columns='region', inplace=True)

    if sub_results_df.size == 0:
        print(f'region {region_name}: no datapoints matching')
    else:
        # save thingy
        sub_results_pq = pa.Table.from_pandas(sub_results_df)
        del sub_results_df

        output_dir = processed / f'{region_name}/boundaries/'
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = output_dir / 'relative.parquet'
        pq.write_table(sub_results_pq, parquet_path)

        print(f'region {region_name}: saved results')


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

    processed = Path(f'/data/cephfs-2/unmirrored/groups/sawitzki/Juno/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    # define variables
    chunks = config['PREPROCESSING'].getfloat('chunks')
    n_roi = config['PREPROCESSING'].getfloat('n_roi')
    overlap = config['PREPROCESSING'].getfloat('overlap')

    pixelsize = config['ImageStats'].getfloat('pixelsize_xy')

    with open(processed / 'sections_px.json', 'r') as f:
        sections_dict = json.load(f)

    # dtype_dict = dict(zip(['transcript_id','overlaps_nucleus','codeword_index'],[np.int64]*3))

    regions = define_regions_to_extract(sections_dict)

    # with mp.Pool(processes=mp.cpu_count()-1) as pool:
    #     parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
    #     with parquet_file.iter_batches() as reader:
    #         results = pool.imap(process_chunk, reader)#, chunksize = int(20000/mp.cpu_count()-1))
    #         pool.close()
    #         pool.join()
    #         results_df = pd.concat(results)
        
        
    #     parquet_file
    # results_df = pd.DataFrame()
    # parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
    # for batch in parquet_file.iter_batches():
    #     batch_df = batch.to_pandas()
    #     processed_df = process_chunk(batch_df)
    #     results_df = pd.concat([results_df, processed_df])
    # for region_name, region_data in regions.items():
    #     save_section(region_name, region_data, results_df) 

    for file in Path(data).glob('*_boundaries.parquet'):
        parquet_file = pq.ParquetFile(file)
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
                regions=regions
            ),
            regions.keys()
        )
        pool.close()
        pool.join()