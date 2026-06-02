from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit


def submit_cmd(config, cmd, section, gpu, double):
    submit_str = f'python submit_sbatch.py -c {config} -d "{cmd}"'
    if section is not None:
        submit_str += f' -s {section}'
    if gpu:
        submit_str += ' -g'
    if double:
        submit_str += ' -d'
    return submit_str

if __name__ == '__main':

    parser = argparse.ArgumentParser(
        prog= 'main', 
        description='''Main file for 10xSegEval. 
            Successively starts image, transcript and boundary processing,
            segmentation algorithms and evaluation.
            '''
    )
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Optional. Path to a config file like "config.toml".'
    )
    parser.add_argument(
        '-s', '--Section',
        default=None,
        help='Optional. Path to dictionary of Sections.'
    )
    args = parser.parse_args()

    config_path = args.Config
    sections = args.Section
    
    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    todo = config['ToDo']
    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']
    methods = config['methods']

    # define paths
    cwd = os.getcwd()
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']
    results = Path(f'{home}/{sample}/results')
    results.mkdir(parents=True, exist_ok=True)
    # add custom section_dictionary to paths
    if section is not None:
        paths['sections_path'] = sections
        with open(config_path, 'wb') as f:
            tomlkit.dump(config, f)

    # preprocess
    if todo['preprocess']:
        if section is None:
            cmd = f'python processing/find_sections.py'
            submit = submit_cmd(cmd=cmd)
            pS = subprocess.Popen(submit, shell=True)
            pS.wait()

        cmd = f'python processing/image_splitting.py'
        submit = submit_cmd(config_path, cmd, section, gpu, double)
        pI = subprocess.Popen(submit, shell=True)

        cmd = f'python processing/transcript_splitting.py'
        submit = submit_cmd(config_path, cmd, section, gpu, double)
        pT = subprocess.Popen(submit, shell=True)

        cmd = f'python processing/boundaries_splitting.py'
        submit = submit_cmd(config_path, cmd, section, gpu, double)
        pB = subprocess.Popen(submit, shell=True)

        pI.wait()
        pT.wait()
        pB.wait()

    if todo['segment']:
        for method in config['methods']:
            pass
        # perhaps function to start a method
        # thus enabling multiple starts with multiprocessing
        # include option to disable
        # since it might be run on a non-cluster system
        # also: argument or config-var for the cluster submition bash
        # no need to have arguments except which file to start
        # since the variables are loaded using the config.ini with lib_ini.sh

    if todo['evaluate']:
        for method in config['methods']:
            