from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit

from submit_sbatch import submit_sbatch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog= 'main',
        description='''Main file for XenSegEval.
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

    tasks = config['Tasks']
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

    tempfile_dir = Path(f'{home}/{sample}/jobs/')
    tempfile_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    # add custom section_dictionary to paths
    if sections is not None:
        paths['sections_path'] = str(sections)
        with open(config_path, 'w') as f:
            tomlkit.dump(config, f)
    # define job arguments
    job_kwargs = config['job']
    job_kwargs.update(
        log_path=str(log_path),
        tempfile_dir=str(tempfile_dir),
        mail=config['owner']['mail'],
    )

    # preprocess
    if tasks['preprocess']:
        print('preprocessing')
        if sections is None:
            cmd = f'python processing/find_sections.py'
            job_kwargs['cmd'] = cmd
            pS = subprocess.Popen(submit_sbatch(job_kwargs), shell=True)
            pS.wait()

        cmd = f'python XenSegEval/processing/image_splitting.py'
        job_kwargs['cmd'] = cmd
        pI = subprocess.Popen(submit_sbatch(job_kwargs), shell=True)
        print('started image splitting.')

        cmd = f'python XenSegEval/processing/transcript_splitting.py'
        job_kwargs['cmd'] = cmd
        pT = subprocess.Popen(submit_sbatch(job_kwargs), shell=True)
        print('started transcript splitting.')

        cmd = f'python XenSegEval/processing/boundaries_splitting.py'
        job_kwargs['cmd'] = cmd
        pB = subprocess.Popen(submit_sbatch(job_kwargs), shell=True)
        print('started boundary splitting.')

        pI.wait()
        pT.wait()
        pB.wait()

    if tasks['segment']:
        print('started segmenting')
        seg = []
        for method in config['methods']:
            # if method in ['proseg', 'ucs']:
            cmd = f'bash XenSegEval/start/{method}.sh'
            if method != 'proseg':
                gpu = True
            job_kwargs['cmd'] = cmd
            seg.append(
                subprocess.Popen(
                    submit_sbatch(job_kwargs, gpu=gpu),
                    shell=True
                )
            )
        for p in seg:
            p.wait() 

    if tasks['evaluate']:
        print('started evaluating')
        evl = []
        for method in config['methods']:
            cmd = f'python XenSegEval/eval/eval.py -c {config_path} -m {method}'
            job_kwargs['cmd'] = cmd
            evl.append(
                subprocess.Popen(
                    submit_sbatch(job_kwargs),
                    shell=True
                )
            )
        for p in evl:
            p.wait()
    
    print('done :3')
