from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit

from 10xSegEval.submit_sbatch import submit_sbatch

def submit_cmd(cmd, config='config.toml', gpu=False, double=False):
    submit_str = f'python submit_sbatch.py -c {config} -m "{cmd}"'
    #if section is not None:
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

    tasks = config['tasks']
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
    log_path = Path(f'{home}/sample/run/logs/{name}/')
    log_path.mkdir(parents=True, exist_ok=True)
    # add custom section_dictionary to paths
    if sections is not None:
        paths['sections_path'] = sections
        with open(config_path, 'w') as f:
            tomlkit.dump(config, f)
    # define job arguments
    job_kwargs = config['job']
    job_kwargs. update(
        log_path=log_path,
        mail=config['owner']['mail'],
    )

    # preprocess
    if tasks['preprocess']:
        print('preprocessing')
        if sections is None:
            cmd = f'python processing/find_sections.py'
            #submit = submit_cmd(cmd=cmd)
            job_kwargs['cmd'] = cmd
            pS = subprocess.Popen(submit_sbatch(job_kwargs), shell=True)
            pS.wait()

        cmd = f'python processing/image_splitting.py'
        #submit = submit_cmd(cmd, double=True)
        job_kwargs['cmd'] = cmd
        pI = subprocess.Popen(submit_sbatch(job_kwargs, double=True), shell=True)
        print('started image splitting.')

        cmd = f'python processing/transcript_splitting.py'
        #submit = submit_cmd(cmd, double=True)
        job_kwargs['cmd'] = cmd
        pT = subprocess.Popen(submit_sbatch(job_kwargs, double=True), shell=True)
        print('started transcript splitting.')

        cmd = f'python processing/boundaries_splitting.py'
        #submit = submit_cmd(cmd, double=True)
        job_kwargs['cmd'] = cmd
        pB = subprocess.Popen(submit_sbatch(job_kwargs, double=True), shell=True)
        print('started boundary splitting.')

        pI.wait()
        pT.wait()
        pB.wait()

    if tasks['segment']:
        print('started segmenting')
        seg = []
        for method in config['methods']:
            # if method in ['proseg', 'ucs']:
            cmd = f'source start/{method}.sh'
            if method != 'proseg':
                gpu = True
            #submit = submit_cmd(cmd=cmd, gpu=gpu, double=True)
            job_kwargs['cmd'] = cmd
            seg.append(
                subprocess.Popen(
                    submit_sbatch(job_kwargs, gpu=gpu, double=True),
                    shell=True
                )
            )
        for p in seg:
            p.wait() 

    if tasks['evaluate']:
        print('started evaluating')
        evl = []
        for method in config['methods']:
            cmd = f'python eval/eval.py -c {config_path} -m {method}'
            #submit = submit_cmd(cmd=cmd)
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