from stardist.models import StarDist2D
from csbdeep.utils import normalize
from tifffile import imread
from pathlib import Path
import numpy as np

from pathlib import Path
import configparser
import argparse
import tomllib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config',
        help='Path to the config file.'
    )
    parser.add_argument('-s', '--Section',
        help='Number of sample-section to segment. Or "all".'
    )
    args = parser.parse_args()

    config_path = args.Config
    sections = args.Section
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    home = config['paths']['home']
    data = Path(config['paths']['data_path'])
    sample = config['paths']['name']
    planes = config['processing']['planes']

    processed = Path(f'{home}{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    if section in range(preprocessing['n_roi']):
        sections = [int(section)]
    else:
        with open(processed / 'sections_px.json') as f:
            section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    for section in sections:
        single_layer = Path(processed / f'{section}/morphology/single_layer/')
        for l, layer in enumerate(single_layer.glob('*0*')):
            p = planes[l]
            for q, quater in enumerate(layer.glob('quatered/q0*.tif')):
                img = imread(quater)
                labels, _ = model.predict_instances(normalize(img))
                np.save(
                    f'{sample}/results/stardist/{section}/layer0{p}/q0{q}.npy',
                    labels
                )
                