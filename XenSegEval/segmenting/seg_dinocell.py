from XenSegEval.utils import get_config_args

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from tifffile import imwrite

import tomlkit
import json

from dinocell import segment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DINOCell')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'dinocell')
    globals().update(variables)

    # load sections
    sections = section_dictionary.keys()

    for section in sections:
        img_path = Path(
            processed / f'{section}/morphology/multi_layer/'
            'morphology.ome.tif'
        )
        output = segment(img_path)

        output_dir = Path(results / f'{section}')
        output_dir.mkdir(parents=True, exist_ok=True)

        imwrite(output_dir / 'prediction.tif', output)
        np.save(output_dir / 'prediction.npy', output)
