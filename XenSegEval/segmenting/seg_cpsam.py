import os
import argparse
from pathlib import Path

from cellpose import models, io
from tifffile import TiffFile, imwrite

import numpy as np

import tomlkit
import json

from XenSegEval.utils import get_config_args

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
    globals().update()

    # preprocessing = config['preprocessing']
    # paths = config['paths']
    # imagestats = config['ImageStats']

    # home = paths['home']
    # sample = paths['sample_name']
    # ## define processed and results directory
    # processed = Path(f'{home}/{sample}/processed/')
    # results = Path(f'{home}/{sample}/results/cpsam')
    # results.mkdir(parents=True, exist_ok=True)
    # ## define sections_dictionary path
    # if 'sections_path' in paths:
    #     sections_path = paths['sections_path']
    # else:
    #     sections_path = processed / 'sections_px.json'

    # pixelsize = imagestats['pixelsize_xy']

    # # load sections_dictionary
    # with open(sections_path) as f:
    #     section_dictionary = json.load(f)
    #     sections = section_dictionary.keys()
    
    # cpsam_model = config['methods']['cpsam']['model']
    # cpsam_eval = config['methods']['cpsam']['eval']

    cpsam_model = method['model']
    cpsam_eval = method['eval']

    io.logger_setup()

    model = models.CellposeModel(**cpsam_model)

    for section in sections:
        multi_layer_quater = Path(processed / f'{section}/morphology/multi_layer/quatered')
        for q, quater in enumerate(multi_layer_quater.glob('q0*.ome.tif')):
            with TiffFile(quater) as tif:
                img = tif.pages[0].asarray()

                masks, flows, styles = model.eval(img, **cpsam_eval)

                res = np.array({'masks': masks, 'flows': flows})

                np.save(
                    f'{sample}/results/cpsam/{section}/q0{q}.npy',
                    res
                )
                