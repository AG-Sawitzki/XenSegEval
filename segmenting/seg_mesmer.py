import os
import tomllib

from pathlib import Path
from tifffile import TiffFile, imwrite
from matplotlib import pyplot as plt
from deepcell.applications import Mesmer

from skimage.segmentation import find_boundaries
import numpy as np
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config
    labels_path = args.Labels

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']
    mesmer_config = config['methods.mesmer']
    os.environ.update({'DEEPCELL_ACCESS_TOKEN': mesmer_config['token']})

    home = paths['home']
    sample = paths['sample_name']
    processed = Path(f'{home}/{sample}/processed/')
    results = Path(f'{home}/{sample}/results/mesmer')
    results.mkdir(parents=True, exist_ok=True)

    with open(processed/'sections_px.json') as f:
            section_dictionary = json.load(f)

    app = Mesmer()
    
    identifiers = ['mt', 'mem', 'ribo']

    for section in section_dictionary.keys():
        with TiffFile(processed / f'{section}/morphology/focus/focus.ome.tif') as tif:
            focus = tf.pages[0].asarray

            # add an empty membrane channel
            focus_mt = np.concatenate(
                (focus[...,0], np.zeros(focus[...,0].shape)),
                axis=2
            )
            focus_mt = np.expand_dims(focus_mt, axis=-1)
            print(focus_mt.shape)
        
            # add ATP1A1/E-Cadherin/CD45 channel
            focus_mem = np.expand_dims(focus[...,0:2], axis=-1)
            print(focus_mem.shape)
        
            # add 18s channel
            focus_ribo = np.expand_dims(focus[...,0:3:2], axis=-1)
            print(focus_ribo.shape)

            del focus

        # predict
        for i, img in enumerate([focus_mt, focus_mem, focus_ribo]):
            identifier = identifiers[i]
            predictions_nuc = app.predict(
                img,
                image_mpp=0.2125,
                compartment='nuclear')
            prediction_cell = app.predict(
                img,
                image_mpp=0.2125,
                compartment='whole-cell'
            )
            # save prediction
            res = np.vstack((prediction_cell, predictions_nuc))
            np.save(
                results / f'output/{section}/prediction_{identifier}.npy',
                res
            )
            imwrite(
                results / f'output/{section}/prediction_{identifier}.tif',
                res
            )