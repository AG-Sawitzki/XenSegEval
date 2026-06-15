from pathlib import Path
import configparser
import argparse
import sys
import os

from stardist.models import StarDist2D
from stardist.models import StarDist3D
from csbdeep.utils import normalize
from tifffile import imread, imwrite
from pathlib import Path
import numpy as np

import json
import tomlkit

from XenSegEval.utils import get_config_args


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

    # load sections_dictionary
    with open(sections_path) as f:
        section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    # creates a pretrained model
    #model = StarDist2D.from_pretrained('2D_versatile_fluo')
    model = StarDist3D.from_pretrained('3D_demo')

    # loop through sections/quaters and segment each
    for section in sections:
        #single_layer = Path(processed / f'{section}/morphology/single_layer/')
        img_path = Path(processed / f'{section}/morphology/multi_layer/morphology.ome.tif')
        #for l, layer in enumerate(single_layer.glob('*0*')):
        #    p = planes[l]
        #    for q, quater in enumerate(layer.glob('quatered/q0*.tif')):
        #        img = imread(quater)
        #        labels, _ = model.predict_instances(normalize(img))
        #        np.save(
        #            f'{sample}/results/stardist/{section}/layer0{p}/q0{q}.npy',
        #            labels
        #        )
        img = imread(img_path)
        labels, _ = model.predict_instances(normalize(img))

        output_dir = Path(results / f'{section}')
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            output_dir / 'prediction.npy',
            labels
        )
        imwrite(
            output_dir / 'prediction.tif',
            labels
        )