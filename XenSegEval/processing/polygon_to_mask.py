from XenSegEval.utils import(
    get_config_args
)
from XenSegEval.processing.utils import (
    wrap_ptm
)

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile
import tomlkit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='boundaries')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-m', '--Method',
        help='Method to convert from polygon(.geojson) to mask(.tif).'
    )

    args = parser.parse_args()

    config_path = args.Config
    method = args.Method

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'boundaries')
    globals().update(variables)

    path = Path(f'{results}/{method}/')
    files = list(path.rglob('*.geojson*'))

    processed = Path(f'{results}').parent / 'processed'
    _section_ = list(sections)[0]
    file = Path(f'{processed}/{_section_}/morphology/focus/focus.ome.tif')
    img = tifffile.imread(file)
    shape = img.shape[:2]

    for file in files:
        if 'cell_' in str(file):
            mode = 'cell'
        elif 'nucleus_' in str(file):
            mode = 'nucleus'
        else:
            mode = None
        wrap_ptm(
            file,
            file.parent,
            shape=shape,
            mode=mode
        )