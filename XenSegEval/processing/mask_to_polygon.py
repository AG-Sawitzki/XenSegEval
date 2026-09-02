from XenSegEval.utils import get_config_args
from XenSegEval.processing.utils import (
    process_roi
)

from pathlib import Path
import argparse

import tifffile
import tomlkit
import numpy as np


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

    path = Path(f'{results}/{method}/output/')
    files = list(path.rglob('prediction*.tif'))

    processed = Path(f'{results}').parent / 'processed'
    _section_ = list(sections)[0]
    # file = Path(f'{processed}/{_section_}/morphology/focus/focus.ome.tif')
    # img = tifffile.imread(file)
    # shape = img.shape[:2]

    for file in files:
        stem = file.stem
        name = ''.join(['polygons', stem[stem.rfind('_'):]])
        name = '.'.join([name, 'geojson'])
        print(name)
        print(file.parent)
        if method == 'mesmer':
            # for i, mode in enumerate(['cell', 'nucleus']):
            img = tifffile.imread(file)[0, ...]
            img = np.squeeze(img)
                # name = '_'.join([name, mode])
        else:
            img = tifffile.imread(file)
        print(img.shape)
        process_roi(
            img,
            file.parent / name,
        )
