from pathlib import Path
import configparser
import argparse

import json
import gzip

import pyarrow.parquet as pq
import pyarrow.Table as pT
import pandas as pd
import numpy as np

def process_chunk(df):
    # df = df.to_pandas()

    regions_mapping = pd.Series(index=df.index, dtype=str).fillna("")

    for region_name, region_data in regions.items():
        y_min = region_data["y_min"]
        x_min = region_data["x_min"]
        y_max = region_data["y_max"]
        x_max = region_data["x_max"]

        regions_mapping[
            (x_min <= df["x_location"])
            & (df["x_location"] <= x_max)
            & (y_min <= df["y_location"])
            & (df["y_location"] <= y_max)
        ] = region_name

    # print(list(set(regions_mapping)))

    df["region"] = regions_mapping

    return df

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
    chunks = config['DEFAULT'].getfloat('chunks')
    min_size = config['DEFAULT'].getfloat('min_size')
    n_roi = config['DEFAULT'].getfloat('n_roi')
    overlap = config['DEFAULT'].getfloat('overlap')
    pixelsize = config['DEFAULT'].getfloat('pixelsize')
    rf = config['DEFAULT'].getfloat('rf')

    with open(processed / 'sections_px.json', 'r') as f:
        sections_dict = json.load(f)

    dtype_dict = dict(zip(['transcript_id','overlaps_nucleus','codeword_index'],[np.int64]*3))

    bins = make_bins(sections_dict)
    print(bins)

    # with mp.Pool(processes=mp.cpu_count()-1) as pool:
    #     parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
    #     with parquet_file.iter_batches() as reader:
    #         results = pool.imap(process_chunk, reader)#, chunksize = int(20000/mp.cpu_count()-1))
    #         pool.close()
    #         pool.join()
    #         results_df = pd.concat(results)
        

    parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
        for batch in parquet_file.iter_batches():
            batch_df = batch.to_pandas()
            processed = process_chunk(batch_df)
            results_df = pd.concat([results_df, processed])
        results_df = results_df.astype(dtype_dict)
        print(results_df.head(5))
        #---save section_absolute---
        results_pq = pT.from_pandas(results_df)
        results_df.to_parquet(path / '{0}/{1}_{0}_absolute.parquet'.format(section, typus), index = False)
