from XenSegEval.utils import get_config_args

import os
import sys
import argparse

from pathlib import Path
from tifffile import TiffFile, imwrite
from deepcell.applications import Mesmer

from skimage.segmentation import find_boundaries
from tomlkit import load
import numpy as np
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DeepCell')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = load(f)

    variables = get_config_args(config, 'mesmer')
    globals().update(variables)

    # load sections
    sections = section_dictionary.keys()

    app = Mesmer()

    # for an empty membrane channel, using the membrane stain,
    # or the ribosome stain
    identifiers = ['mt', 'mem', 'ribo']

    for section in sections:
        with TiffFile(
            processed /
            f'{section}/morphology/focus/focus.ome.tif'
        ) as tif:
            focus = tif.pages[0].asarray()
            # print(focus.shape)

            # add an empty membrane channel
            focus_mt = np.expand_dims(focus[..., 0], axis=(0, -1))
            focus_mt = np.concatenate(
                (focus_mt, np.zeros(focus_mt.shape)),
                axis=-1
            )
            # print(focus_mt.shape)

            # add ATP1A1/E-Cadherin/CD45 channel
            focus_mem = np.expand_dims(focus[..., 0:2], axis=0)
            print(focus_mem.shape)
            print(sum(sum(focus_mem)))

            # add 18s channel
            focus_ribo = np.expand_dims(focus[..., 0:3:2], axis=0)
            # print(focus_ribo.shape)

        # predict
        for i, img in enumerate([focus_mt, focus_mem, focus_ribo]):
            identifier = identifiers[i]
            predictions_nuc = app.predict(
                img,
                image_mpp=pixelsizeXY,
                compartment='nuclear')
            prediction_cell = app.predict(
                img,
                image_mpp=pixelsizeXY,
                compartment='whole-cell'
            )
            # save prediction
            res = np.vstack((prediction_cell, predictions_nuc))

            output_dir = Path(results / f'{section}')
            output_dir.mkdir(parents=True, exist_ok=True)

            np.save(
                output_dir / f'prediction_{identifier}.npy',
                res
            )
            imwrite(
                output_dir / f'prediction_{identifier}.tif',
                res
            )
