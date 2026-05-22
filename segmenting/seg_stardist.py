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
    try:
        sections = [int(sections)]
    except:
        sections = range(12)
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    home = config['paths']['home']
    data = Path(config['paths']['data_path'])
    sample = config['paths']['name']
    planes = config['processing']['planes']

    processed = Path(f'{home}{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    single_layer = Path(processed / 'morphology/single_layer/')

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    for section in sections:
        for l, layer in enumerate(single_layer.glob('*0*')):
            l = planes[l]
            for q, quater in enumerate(layer.glob('quatered/q0*.tif')):

                img = imread(quater)

                labels, _ = model.predict_instances(normalize(img))

                np.save(
                    f'{sample}/results/stardist/{section}/layer0{l}/q0{q}.npy',
                    labels
                )