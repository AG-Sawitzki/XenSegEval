from XenSegEval.utils import get_config_args 
from XenSegEval.plotting.utils import polygon_overlay

import gzip
import pickle
import argparse
from pathlib import Path

import tomlkit
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Ovrl.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-m', '--Method',
        help='Method to evaluate.'
    )
    parser.add_argument(
        '-s', '--Section',
        help='Section segmented.'
    )

    args = parser.parse_args()

    method = args.Method
    section = args.Section
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'plot')
    globals().update(variables)

    img = f'{processed}/{section}/morphology/focus/focus.ome.tif'

    mask_path = Path(f'{results}/{method}/output/{section}/')

    files = sorted(Path(mask_path).glob(f'*.geojson*'))

    gdfs = []
    labels = []

    if method == 'proseg':
        for file in files:
            with gzip.open(file) as f:
                gdf = gpd.read_file(f)
            if 'layer' in gdf.columns:
                for layer in set(gdf['layer']):
                    gdfs.append(gdf[gdf['layer']==layer])
                    labels.append(file.stem+str(layer))
            else:
                gdfs.append(gdf)
                labels.append(file.stem+'all')
    else:
        for file in files:
            gdfs.append(gpd.read_file(file))
            labels.append(file.stem)

    for i, gdf in enumerate(gdfs):
        output_path = Path(f'{results}/{method}/visualisation/')
        output_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        polygon_overlay(
            gdf, img,
            Path(output_path) / f'outline_{method}_{labels[i]}.png',
            fig, ax,
            pixelsize_xy=pixelsizeXY
        )