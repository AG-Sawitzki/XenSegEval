import os
import argparse
from pathlib import Path

from cellpose import models, io
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

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']
    #mesmer_config = config['methods.mesmer']
    #os.environ.update({'DEEPCELL_ACCESS_TOKEN': mesmer_config['token']})

    home = paths['home']
    sample = paths['sample_name']
    processed = Path(f'{home}/{sample}/processed/')
    results = Path(f'{home}/{sample}/results/cpsam')
    results.mkdir(parents=True, exist_ok=True)

    pixelsize = imagestats['pixelsize_xy']

    if section in range(preprocessing['n_roi']):
        sections = [int(section)]
    else:
        with open(processed / 'sections_px.json') as f:
            section_dictionary = json.load(f)
        sections = section_dictionary.keys()
    
    cpsam_model = config['methods']['cpsam']['model']
    cpsam_eval = config['methods']['cpsam']['eval']

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
                