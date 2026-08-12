from XenSegEval.utils import get_config_args, mask_to_polygons
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

    for file in sorted(Path(mask_path).glob(f'*.geojson*')):
        output_path = Path(f'{results}/{method}/visualisation/')
        output_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        polygon_overlay(
            gdf, img,
            Path(output_path) / f'outline_{method}_{file.stem}.png',
            fig, ax,
            pixelsize_xy=pixelsizeXY
        )
