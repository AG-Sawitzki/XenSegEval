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
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    parser.add_argument('-s', '--Section', help='Section segmented.')
    args = parser.parse_args()

    method = args.Method
    section = args.Section
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    img = f'{processed}/{section}/morphology/focus/focus.ome.tif'

    mask_path = Path(f'{results}/{method}/output/{section}/')

    if method in ['proseg', 'cpsam']:
        ext = 'tif'
    else:
        ext = 'npy'

    for file in sorted(Path(mask_path).glob(f'prediction*.{ext}')):
        polygons = mask_path / f'polygons_{file.stem}.geojson'

        if not polygons.is_file() and method != 'proseg':
            if method == 'cpsam':
                mask = tifffile.imread(file)
            else:
                # try:
                mask = np.load(file, allow_pickle=True)
                # except:
                #     try:
                #         mask = np.load(file)
                #     except:
                #         mask = pickle.load(file)
                #         mask = np.array(mask)

            if method == 'mesmer':
                mask = np.squeeze(mask[0, ...])

            polygons = mask_to_polygons(
                mask,
                polygons
            )

        if method == 'proseg':
            if '_' in str(file):
                with gzip.open(
                    file.with_name(
                        'cell-polygons_layers.geojson.gz'
                    )
                ) as f:
                    polygons = gpd.read_file(f)
                    layer = int(file.stem[-1:])
                    polygons = polygons[polygons['layer'] == layer]
            else:
                with gzip.open(
                    file.with_name(
                        'cell-polygons.geojson.gz'
                    )
                ) as f:
                    polygons = gpd.read_file(f)

        output_path = Path(f'{results}/{method}/visualisation/')
        output_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        polygon_overlay(
            polygons, img,
            Path(output_path) / f'outline_{method}_{file.stem}.png',
            fig, ax
        )
