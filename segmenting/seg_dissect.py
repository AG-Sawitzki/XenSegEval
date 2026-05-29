import os
import argparse

import dissect
from pathlib import Path
from tifffile import TiffFile, imwrite

from skimage.segmentation import find_boundaries
from tomli import load
import numpy as np
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-s', '--Section', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config
    section = args.Section

    with open(config_path, 'rb') as f:
        config = load(f)

    dissect_parameters = config['methods.dissect']
    imagestats = config['ImageStats']
    paths = config['paths']
    preprocessing = config['preprocessing']

    pixelsize = imagestats['pixelsize_xy']

    home = paths['home']
    sample = paths['sample_name']
    processed = Path(f'{home}/{sample}/processed/')

    results = Path(f'{home}/{sample}/results/dissect')
    results.mkdir(parents=True, exist_ok=True)


    if section in range(preprocessing['n_roi']):
        sections = [int(section)]
    else:
        with open(processed / 'sections_px.json') as f:
            section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    for section in sections:
        mask = dissect.segmentation(
            img_path=Path(processed / f'{section}/morphology/focus/focus.ome.tif'),
            platform='xenium',
            gene_mtx_filename=Path(processed / f'{section}/transcripts/relative.csv'),
            config_file=Path('segmenting/dissect_config.yaml'),
            weights_file=Path('segmenting/dissect_weights.pth'),
            output=Path(results / f'output/{section}')
        )
