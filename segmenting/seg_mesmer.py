import os
import argparse

from pathlib import Path
from tifffile import TiffFile, imwrite
from deepcell.applications import Mesmer

from skimage.segmentation import find_boundaries
from tomlkit import load
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    
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
    results = Path(f'{home}/{sample}/results/mesmer')
    results.mkdir(parents=True, exist_ok=True)
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = paths['sections_path']
    else:
        sections_path = processed / 'sections_px.json'

    # load sections_dictionary
    with open(sections_path) as f:
        section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    pixelsize = imagestats['pixelsize_xy']

    app = Mesmer()

    # for an empty membrane channel, using the membrane stain,
    # or the ribosome stain
    identifiers = ['mt', 'mem', 'ribo']

    for section in sections:
        with TiffFile(processed / f'{section}/morphology/focus/focus.ome.tif') as tif:
            focus = tif.pages[0].asarray()
            #print(focus.shape)

            # add an empty membrane channel
            focus_mt = np.expand_dims(focus[...,0], axis=(0,-1))
            focus_mt = np.concatenate(
                (focus_mt, np.zeros(focus_mt.shape)),
                axis=-1
            )
            #print(focus_mt.shape)
        
            # add ATP1A1/E-Cadherin/CD45 channel
            focus_mem = np.expand_dims(focus[...,0:2], axis=0)
            #print(focus_mem.shape)
        
            # add 18s channel
            focus_ribo = np.expand_dims(focus[...,0:3:2], axis=0)
            #print(focus_ribo.shape)

            del focus

        # predict
        for i, img in enumerate([focus_mt, focus_mem, focus_ribo]):
            identifier = identifiers[i]
            predictions_nuc = app.predict(
                img,
                image_mpp=pixelsize,
                compartment='nuclear')
            prediction_cell = app.predict(
                img,
                image_mpp=pixelsize,
                compartment='whole-cell'
            )
            # save prediction
            res = np.vstack((prediction_cell, predictions_nuc))

            res_path = Path(results / f'output/{section}/')
            res_path.mkdir(parents=True, exist_ok=True)

            np.save(
                results / f'output/{section}/prediction_{identifier}.npy',
                res
            )
            imwrite(
                results / f'output/{section}/prediction_{identifier}.tif',
                res
            )