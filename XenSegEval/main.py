from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit

from XenSegEval.utils import get_config_args, submit_sbatch

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

    if sections is not None:
        config['paths']['sections_path'] = str(sections)
        with open(config_path, 'w') as f:
            tomlkit.dump(config, f)

    variables = get_config_args(config, 'main')
    globals().update(variables)
    
    gpu = sbatch_kwargs['gpu']

    # preprocess
    if tasks['preprocess']:
        del sbatch_kwargs['gpu']
        print('preprocessing')
        if sections is None:
            cmd = f'pixi run python -m XenSegEval.processing.find_sections -c {config_path}'
            sbatch_kwargs['cmd'] = cmd
            pS = subprocess.Popen(submit_sbatch(**sbatch_kwargs))
            pS.wait()

        cmd = f'pixi run python -m XenSegEval.processing.image_splitting -c {config_path}'
        sbatch_kwargs['cmd'] = cmd
        pI = subprocess.Popen(submit_sbatch(**sbatch_kwargs))
        print('started image splitting.')

        cmd = f'pixi run python -m XenSegEval.processing.transcript_splitting -c {config_path}'
        sbatch_kwargs['cmd'] = cmd
        pT = subprocess.Popen(submit_sbatch(**sbatch_kwargs))
        print('started transcript splitting.')

        cmd = f'pixi run python -m XenSegEval.processing.boundaries_splitting -c {config_path}'
        sbatch_kwargs['cmd'] = cmd
        pB = subprocess.Popen(submit_sbatch(**sbatch_kwargs))
        print('started boundary splitting.')

        pI.wait()
        pT.wait()
        pB.wait()

    if tasks['segment']:
        print(config['sbatch'])
        sbatch_kwargs['gpu'] = gpu
        print('started segmenting')
        seg = []
        for method in config['methods']:
            cmd = f'bash XenSegEval/start/{method}.sh {config_path}'
            sbatch_kwargs['cmd'] = cmd
            seg.append(subprocess.Popen(submit_sbatch(**sbatch_kwargs)))
        for p in seg:
            p.wait() 

    if tasks['evaluate']:
        print('started evaluating')
        evl = []
        for method in config['methods']:
            cmd = f'''pixi run -e eval \
                python -m XenSegEval.eval.eval \
                -c {config_path} -m {method}
            '''
            sbatch_kwargs['cmd'] = cmd
            evl.append(subprocess.Popen(submit_sbatch(**sbatch_kwargs)))
        for p in evl:
            p.wait()
    
    print('done :3')
