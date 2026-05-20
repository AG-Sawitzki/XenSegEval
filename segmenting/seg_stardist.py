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
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    home = config['paths']['home']
    data = Path(config['paths']['data_path'])
    sample = config['paths']['name']

    processed = Path(f'{home}{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)
    path = Path('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/processed/data_processed/image-data_processed/morphology/36/quatered')


    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    for i in range(4):
        img = imread(path / 'morphology_L6-7_q0{0}.tif'.format(i))

        labels, _ = model.predict_instances(normalize(img))
        
        np.save(f'{sample}...results/stardist/36_q0{0}.npy'.format(i), labels)