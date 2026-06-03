from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit


def submit_cmd(cmd, config='config.toml' gpu=False, double=False):
    submit_str = f'python submit_sbatch.py -c {config} -d "{cmd}"'
    #f section is not None:
    #    submit_str += f' -s {section}'
    if gpu:
        submit_str += ' -g'
    if double:
        submit_str += ' -d'
    return submit_str


if __name__ == '__main__':

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
    if sections is not None:
        paths['sections_path'] = sections
        with open(config_path, 'w') as f:
            tomlkit.dump(config, f)

    # preprocess
    if todo['preprocess']:
        print('preprocessing')
        if sections is None:
            cmd = f'python processing/find_sections.py'
            submit = submit_cmd(cmd=cmd)
            pS = subprocess.Popen(submit, shell=True)
            pS.wait()

        cmd = f'python processing/image_splitting.py'
        submit = submit_cmd(config_path, cmd, sections, gpu, double)
        pI = subprocess.Popen(submit, shell=True)
        print('started image splitting.')

        cmd = f'python processing/transcript_splitting.py'
        submit = submit_cmd(config_path, cmd, sections, gpu, double)
        pT = subprocess.Popen(submit, shell=True)
        print('started transcript splitting.')

        cmd = f'python processing/boundaries_splitting.py'
        submit = submit_cmd(config_path, cmd, sections, gpu, double)
        pB = subprocess.Popen(submit, shell=True)
        print('started boundary splitting.')

        pI.wait()
        pT.wait()
        pB.wait()

    if todo['segment']:
        print('started segmenting')
        seg = []
        for method in config['methods']:
            # if method in ['proseg', 'ucs']:
            cmd = f'source start/{method}.sh'
            if method != 'proseg':
                gpu = True
            submit = submit_cmd(cmd=cmd, gpu=gpu, double=True)
            seg.append(subprocess.Popen(submit, shell=True))
        for p in seg:
            p.wait() 

    if todo['evaluate']:
        print('started evaluating')
        evl = []
        for method in config['methods']:
            cmd = f'python eval/eval.py -c {config_path} -m {method}'
            submit = submit_cmd(cmd=cmd)
            evl.append(subprocess.Popen(submit, shell=True))
        for p in evl:
            p.wait()
    
    print('done :3')