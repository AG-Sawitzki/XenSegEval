from XenSegEval.utils import get_config_args

import os
import sys
import argparse
from pathlib import Path

from cellpose import models, io
from tifffile import TiffFile, imwrite, imread

import numpy as np

import tomlkit
import json

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

    io.logger_setup()

    model = models.CellposeModel(**cpsam_model)

    for section in sections:
        img_path = Path(
            processed /
            f'{section}/morphology/multi_layer/morphology.ome.tif'
        )
        # multi_layer_quater = Path(processed / f'{section}/morphology/multi_layer/quatered')
        # for q, quater in enumerate(multi_layer_quater.glob('q0*.ome.tif')):
        #    with TiffFile(quater) as tif:
        #        img = tif.pages[0].asarray()

        #        masks, flows, styles = model.eval(img, **cpsam_eval)

        #        res = np.array({'masks': masks, 'flows': flows})

        #        np.save(
        #            f'{sample_name}/results/cpsam/{section}/q0{q}.npy',
        #            res
        #        )
        # with TiffFile(img_path) as tif:
        img = imread(img_path)
        masks, flows, styles = model.eval(img, **cpsam_eval)
        prediction = np.array({'masks': masks, 'flows': flows})
        output_dir = results / f'{section}'
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / 'output.npy',
            prediction
        )
        imwrite(
            output_dir / f'output.tif',
            masks
        )
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
