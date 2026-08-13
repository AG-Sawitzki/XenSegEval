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
    identifiers = ['mt', 'mem', 'ribo', 'cyto']

    for section in sections:
        with TiffFile(
            processed /
            f'{section}/morphology/focus/focus.ome.tif'
        ) as tif:
            focus = tif.pages[0].asarray()
            dapi = focus[...,0]
            dapi_exp = np.expand_dims(dapi, axis=0)
            zeros = np.zeros((1,)+dapi.shape)
            focus_mt = np.concatenate((dapi_exp, zeros))
            focus_mt_exp = np.expand_dims(
                np.moveaxis(focus_mt, 0, -1),
                axis=0
            )
            imgs = [focus_mt_exp]
            for channel in range(1,min(focus.shape)):
                focus_c = focus[...,channel]
                focus_c_exp = np.expand_dims(focus_c, axis=0)
                dapi_mem = np.concatenate(
                    (dapi_exp, focus_c_exp)
                )
                dapi_mem_exp = np.expand_dims(
                    np.moveaxis(dapi_mem, 0, -1),
                    axis=0
                )
                imgs.append(dapi_mem_exp)
        # predict
        for i, img in enumerate(imgs):
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
                res,
                allow_pickle=True
            )
            imwrite(
                output_dir / f'prediction_{identifier}.tif',
                res
            )
