import os
import argparse
from pathlib import Path

import numpy as np
from tifffile import imwrite

import tomlkit
import json

from dinocell import segment


if __name__ == '__main__':
    with open('config.toml', 'rb') as f:
         config = tomlkit.load(f)

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']

    home = paths['home']
    sample = paths['sample_name']
    ## define processed and results directory
    processed = Path(f'{home}/{sample}/processed/')
    results = Path(f'{home}/{sample}/results/dinocell')
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
        img_path = Path(processed / f'morphology/multi_layer/morphology.tif')

        output = segment(img_path)

        imwrite(Path(results / f'{section}/output.tif'), output)
        np.save(Path(results / f'{section}/output.npy'), output)
