from pathlib import Path
import configparser
import subprocess
import argparse
import tomllib
import os

if __name__ == '__main':

    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    preprocessing = config['precossing']
    paths = config['paths']
    imagestats = config['ImageStats']
    methods = config['methods']

    # define paths
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']
    
    results = Path(f'{home}/{sample}/results')
    results.mkdir(parents=True, exist_ok=True)

    # preprocess
    cmd = f'python processing/image_splitting.py -c {config_path}'
    os.system(cmd)

    cmd = f'python processing/transcript_splitting.py -c {config_path}'
    os.system(cmd)

    cmd = f'python processing/boundaries_splitting.py -c {config_path}'
    os.system(cmd)


    if 'CPSAM' in config.sections():
        path = 
        subprocess.run(path,)

    for method in methods:
        path = f'bash start/{method}'
        os.system(cmd)
        #subprocess.run()

        # perhaps function to start a method
        # thus enabling multiple starts with multiprocessing
        # include option to disable
        # since it might be run on a non-cluster system
        # also: argument or config-var for the cluster submition bash
        # no need to have arguments except which file to start
        # since the variables are loaded using the config.ini with lib_ini.sh