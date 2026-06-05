import os
import argparse

import dissect
from pathlib import Path
from tifffile import TiffFile, imwrite

from skimage.segmentation import find_boundaries
from tomlkit import load
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DISSECT.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = load(f)

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']

    home = paths['home']
    sample = paths['sample_name']
    ## define processed and results directory
    processed = Path(f'{home}/{sample}/processed/')
    results = Path(f'{home}/{sample}/results/dissect')
    results.mkdir(parents=True, exist_ok=True)
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = paths['sections_path']
    else:
        sections_path = processed / 'sections_px.json'

    pixelsize = imagestats['pixelsize_xy']

    # load sections_dictionary
    with open(sections_path) as f:
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
