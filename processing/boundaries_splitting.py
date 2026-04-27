from pathlib import Path
import configparser
import argparse

import json
import gzip

import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def process_chunk(df):

    y_binned = pd.cut(df['y_location'], bins[0], labels = ['y0', 'y1', 'y2', 'y3'], include_lowest = True).to_numpy()
    x_binned = pd.cut(df['x_location'], bins[1], labels = ['x0', 'x1', 'x2'], include_lowest = True).to_numpy()
    
    index_ = df.index.to_numpy()
    index = pd.MultiIndex.from_arrays([y_binned, x_binned, index_], names = ('y_bin', 'x_bin', 'idx'))
    
    df.index = index
    
    df.reset_index(inplace = True)
    df.dropna(axis = 0, inplace = True)
    df.set_index(['y_bin', 'x_bin', 'idx'], inplace = True)
    df.sort_index(inplace = True)
    
    if df.index.is_monotonic_increasing == True:
        print('sorted')
        return df
    else:
        print('sorting already prob here')
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

    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
        with parquet_file.iter_batches() as reader:
            results = pool.imap(process_chunk, reader)#, chunksize = int(20000/mp.cpu_count()-1))
            pool.close()
            pool.join()
            results_df = pd.concat(results)
        
        
        parquet_file

    parquet_file = pq.ParquetFile(data / 'cell_boundaries.parquet')
        for batch in parquet_file.iter_batches():
            batch_df = batch.to_pandas()
            processed = process_chunk(batch_df)
            results_df = pd.concat([results_df, processed])
        results_df = results_df.astype(dtype_dict)
        print(results_df.head())
        #---save section_absolute---
        results_df.to_parquet(path / '{0}/{1}_{0}_absolute.parquet'.format(section, typus), index = False)
