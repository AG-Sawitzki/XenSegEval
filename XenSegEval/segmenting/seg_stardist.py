from XenSegEval.utils import get_config_args

from pathlib import Path
import configparser
import argparse
import sys
import os

import zarr
from stardist.models import StarDist2D
from stardist.models import StarDist3D
from csbdeep.utils import normalize
from tifffile import imread, imwrite
from pathlib import Path
import numpy as np

import json
import tomlkit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='StarDist')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'stardist')
    globals().update(variables)

    style = method['style']

    # load sections
    sections = section_dictionary.keys()

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # model = StarDist3D.from_pretrained('3D_demo')

    # loop through sections/quaters and segment each
    for section in sections:
        if style == '3D':
            for L, p in enumerate(planes):
                img_path = Path(
                    processed /
                    f'{section}/morphology/'
                    f'single_layer/layer0{p}/morphology.tif'
                )
                img = imread(img_path)
                labels, _ = model.predict_instances(normalize(img))

                output_dir = Path(results / f'{section}')
                output_dir.mkdir(parents=True, exist_ok=True)

                np.save(
                    output_dir / f'prediction_p{p}.npy',
                    labels,
                    allow_pickle=True
                )
                imwrite(
                    output_dir / f'prediction_p{p}.tif',
                    labels
                )
        elif style == 'focus':
            img_path = Path(
                processed /
                f'{section}/morphology/focus/focus.ome.tif'
            )
            img_store = imread(img_path, aszarr=True)
            img_zarr = zarr.open(img_store, mode='r')
            img = img_zarr[...,0]
            labels, _ = model.predict_instances(normalize(img))
            output_dir = Path(results / f'{section}')
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(
                output_dir / f'prediction_focus.npy',
                labels,
                allow_pickle=True
            )
            imwrite(
                output_dir / f'prediction_focus.tif',
                labels
            )