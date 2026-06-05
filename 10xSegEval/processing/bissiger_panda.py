from pathlib import Path
import configparser
import argparse
import functools

import json
import gzip
import tomlkit

import multiprocessing as mp
import pandas as pd
import numpy as np
import cycler
import matplotlib.pyplot as plt

debug_mode = False
parallel_mode = True

def define_regions_to_extract(sections_dict, pixelsize):

    regions = {}
    color_cycler = cycler.cycler(color=["c", "m", "y", "k", "b"])()

    for region, bbox in sections_dict.items():
        y_min_px, x_min_px = bbox[0]
        y_max_px, x_max_px = bbox[1]

        regions[region] = {
            "y_min": y_min_px * pixelsize,
            "x_min": x_min_px * pixelsize,
            "y_max": y_max_px * pixelsize,
            "x_max": x_max_px * pixelsize,
            "style": next(color_cycler),
        }

    if debug_mode:
        for region_name, region_data in regions.items():
            plt.plot(
                [region_data["y_min"], region_data["y_max"]],
                [region_data["x_min"], region_data["x_max"]],
                **region_data["style"],
            )
            plt.plot(
                [region_data["y_max"], region_data["y_min"]],
                [region_data["x_min"], region_data["x_max"]],
                **region_data["style"],
            )
            y_mean = (region_data["y_min"] + region_data["y_max"]) / 2
            x_mean = (region_data["x_min"] + region_data["x_max"]) / 2
            plt.text(y_mean, x_mean, region_name)
        plt.show()

    return regions


def process_chunk(df, regions):

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


def relative(df, region_data):
    df["y_location"] = df["y_location"] - region_data["y_min"]
    df["x_location"] = df["x_location"] - region_data["x_min"]
    return df


def pixelate(df):
    df["y_location"] = (df["y_location"] / pixelsize).round(0).astype(np.int64)
    df["x_location"] = (df["x_location"] / pixelsize).round(0).astype(np.int64)
    return df


def save_section(region_name, df, regions):
    region_data = regions[region_name]

    # main selection
    sub_results_df = df[df["region"] == region_name]
    print(sub_results_df.loc[:10, "x_location":"y_location"])

    # remove region offset
    sub_results_df = relative(sub_results_df, region_data)

    # pixelation
    sub_results_df = pixelate(sub_results_df)

    # cleanup
    sub_results_df.drop(columns="region", inplace=True)

    if sub_results_df.size == 0:
        print(f"region {region_name}: no datapoints matching")

    else:
        # save thingy
        output_dir = processed / f"{region_name}/transcripts/"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "relative.csv"
        sub_results_df.to_csv(csv_path, index=False)

        # compress for ProSeg
        gz_path = output_dir / "relative.csv.gz"
        sub_results_df.to_csv(gz_path, index=False, compression="infer")

        print(f"region {region_name}: saved results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='trans.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.')
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
    pixelsize = pixelsizeXY

    regions = define_regions_to_extract(section_dictionary,pixelsize)
    print(regions)

    print('Processing Chunks.')
    if parallel_mode: 
        print('Processing Chunks (parallel).')
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            with pd.read_csv(
                data / 'transcripts.csv.gz', 
                compression = 'infer', 
                dtype=dtype_dict, 
                chunksize = 20000
            ) as reader:
                results = pool.imap(functools.partial(process_chunk, regions=regions), reader) # chunksize = int(20000/mp.cpu_count()-1))
                pool.close()
                pool.join()
                results_df = pd.concat(results)
        results_df.index = results_df.index.sort_values()
        print(results_df.head(n=5))

        # save the sections (parallel)
        with mp.Pool(processes=mp.cpu_count()-1) as pool:
            pool.imap_unordered(functools.partial(save_section, df=results_df, regions=regions), regions.keys())
            pool.close()
            pool.join()

    else: 
        print("Processing Chunks (non parallel)")
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            with pd.read_csv(
                data / "transcripts.csv.gz",
                compression="infer",
                dtype=dtype_dict,
                chunksize=20000
            ) as reader:
                results = []
                for i, chunk in enumerate(reader):
                    print(i)
                    result = process_chunk(chunk, regions)
                    results.append(result)
                    
                    if debug_mode and i > 100:
                        break
                results_df = pd.concat(results)

        # save the regions (non-parallel)
        for region_name, region_data in regions.items():
            save_section(region_name, results_df, regions)
