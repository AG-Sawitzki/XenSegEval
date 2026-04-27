from pathlib import Path
import configparser
import subprocess
import argparse

if __name__ == '__main':

    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    config = configparser.ConfigParser()
    config.read(config_path)
    
    data = Path(config['PATHS']['data_path'])
    sample = config['PATHS']['sample_name']
    
    results = Path(f'/data/cephfs-2/unmirrored/groups/sawitzki/Juno/{sample}/results')
    results.mkdir(parents=True, exist_ok=True)

    # define variables
    chunks = config['DEFAULT'].getfloat('chunks')
    min_size = config['DEFAULT'].getfloat('min_size')
    n_roi = config['DEFAULT'].getfloat('n_roi')
    overlap = config['DEFAULT'].getfloat('overlap')
    pixelsize = config['DEFAULT'].getfloat('pixelsize')
    rf = config['DEFAULT'].getfloat('rf')

    if 'CPSAM' in config.sections():
        path = config_path[:-len('config.ini')]+'start/cpsam.sh'
        subprocess.run(path,)

    for method in config.sections()[2:]:
        path = config_path[:-len('config.ini')]+f'start/{lower(method)}'
        subprocess.run()

        # perhaps function to start a method
        # thus enabling multiple starts with multiprocessing
        # include option to disable
        # since it might be run on a non-cluster system
        # also: argument or config-var for the cluster submition bash
        # no need to have arguments except which file to start
        # since the variables are loaded using the config.ini with lib_ini.sh