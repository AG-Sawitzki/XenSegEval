from XenSegEval.utils import get_config_args

import os
import sys
import argparse
from pathlib import Path

from cellpose import models, io

import zarr
import tomlkit
import numpy as np
from tifffile import TiffFile, imwrite, imread

print(os.getcwd())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CPSAM.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'cpsam')
    globals().update(variables)

    # load sections
    sections = section_dictionary.keys()

    cpsam_model = method['model']
    cpsam_eval = method['eval']
    style = method['style']

    io.logger_setup()

    model = models.CellposeModel(**cpsam_model)

    for section in sections:
        if style == '3D':
            img_path = Path(
                processed /
                f'{section}/morphology/multi_layer/morphology.ome.tif'
            )
        elif style == 'focus':
            img_path = Path(
                processed /
                f'{section}/morphology/focus/focus.ome.tif'
            )
        else:
            print('No style given. Defaulting to "focus"')
            img_path = Path(
                processed /
                f'{section}/morphology/focus/focus.ome.tif'
            )

        img_store = imread(img_path, aszarr=True)
        img_zarr = zarr.open(img_store, mode='r')
        img = np.array(img_zarr)
        masks, flows, styles = model.eval(img, **cpsam_eval)
        prediction = np.array({'masks': masks, 'flows': flows})
        output_dir = results / f'{section}'
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / 'output.npy',
            prediction
        )
        if style == '3D':
            for L, p in enumerate(planes):
                imwrite(
                    output_dir / f'prediction_p{p}.tif',
                    masks[L, ...]
                )
                np.save(
                    output_dir / f'prediction_p{p}.tif',
                    masks[L, ...],
                    allow_pickle=True
                )
        else:
            np.save(
                output_dir / 'prediction_focus.npy',
                masks[...,0],
                allow_pickle=True
            )
            imwrite(
                output_dir / 'prediction_focus.tif',
                masks[...,0]
            )