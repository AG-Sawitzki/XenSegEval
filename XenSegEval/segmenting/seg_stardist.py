from stardist.models import StarDist2D
from stardist.models import Stardist3D
from csbdeep.utils import normalize
from tifffile import imread
from pathlib import Path
import numpy as np

from pathlib import Path
import configparser
import argparse
import tomllib

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
        config = load(f)

    variables = get_config_args(config, 'main')
    globals().update(variables)

    # # define paths
    # paths = config['paths']
    # home = paths['home']
    # data = Path(config['paths']['data_path'])
    # sample = paths['sample_name']
    # ## define processed directory    
    # processed = Path(f'{home}{sample}/processed')
    # processed.mkdir(parents=True, exist_ok=True)
    # ## define sections_dictionary path
    # if 'sections_path' in paths:
    #     sections_path = paths['sections_path']
    # else:
    #     sections_path = processed / 'sections_px.json'

    # # planes of interest
    # planes = config['preprocessing']['planes']

    # # load sections_dictionary
    # with open(sections_path) as f:
    #     section_dictionary = json.load(f)
    # sections = section_dictionary.keys()

    # creates a pretrained model
    #model = StarDist2D.from_pretrained('2D_versatile_fluo')
    model = Stardist3D.from_pretrained('3D_demo')

    # loop through sections/quaters and segment each
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
                