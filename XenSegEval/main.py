from XenSegEval.utils import get_config_args, submit_sbatch

from pathlib import Path
import configparser
import subprocess
import argparse
import os

import tomlkit

PCA_CAPABLE = [
    # 'cpsam',
    # 'dinocell',
    # 'dissect',
    'mesmer',
    # 'proseg',
    # 'stardist'
]
'''LIST OF CURRENTLY SUPPORTED ALGORITHMS FOR CSEs PCA analysis'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
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
    args = parser.parse_args()

    config_path = args.Config

    if config_path is None:
        cwd = os.getcwd()
        config_path = cwd + '/config.toml'
    print(config_path)

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    if (
        (sections is None or gt_path is None)
        and tasks['evaluate'] is True
    ):
        print(
            'No section and/or ground truth provided for evaluation.'
            ' Please check "[preprocessing]".'
        )

    variables = get_config_args(config, 'main')
    globals().update(variables)

    gpu = sbatch_kwargs['gpu']
    mem = sbatch_kwargs['mem']

    # preprocess
    if tasks['preprocess']:
        del sbatch_kwargs['gpu']
        print('preprocessing')
        if sections is None:
            cmd = (
                'pixi run python -m XenSegEval.processing.find_sections'
                f' -c {config_path}'
            )
            sbatch_kwargs['cmd'] = cmd
            pS = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
            pS.wait()

        cmd = (
            'pixi run python -m XenSegEval.processing.image_splitting'
            f' -c {config_path}'
        )
        sbatch_kwargs['cmd'] = cmd
        pI = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
        print('Started image splitting.')

        cmd = (
            'pixi run python -m XenSegEval.processing.transcript_splitting'
            f' -c {config_path}'
        )
        sbatch_kwargs['cmd'] = cmd
        pT = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
        print('Started transcript splitting.')

        cmd = (
            'pixi run python -m XenSegEval.processing.boundaries_splitting'
            f' -c {config_path}'
        )
        sbatch_kwargs['cmd'] = cmd
        pB = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
        print('Started boundary splitting.')

        pB.wait()

        if include_xenium:
            cmd = (
                'pixi run python -m XenSegEval.processing.prepare_xenium-seg'
                f' -c {config_path}'
            )
            sbatch_kwargs['cmd'] = cmd
            pX = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
            print('Preparing Xenium Boundaries.')

        pI.wait()
        pT.wait()

    if tasks['segment']:
        sbatch_kwargs['gpu'] = gpu
        print('started segmenting')
        seg = []
        for method in config['methods']:
            cmd = f'bash XenSegEval/start/{method}.sh {config_path}'
            if method == 'segger':
                cmd = f'bash XenSegEval/start/{method}.sh {data_path} {results}/{method}/output/'
            sbatch_kwargs['cmd'] = cmd
            if method in ['dissect', 'segger']:
                sbatch_kwargs['mem'] = 128
            seg.append(
                subprocess.Popen(
                    submit_sbatch(**sbatch_kwargs),
                    shell=True
                )
            )
            sbatch_kwargs['mem'] = mem
            sbatch_kwargs['gpu'] = gpu
        for p in seg:
            p.wait()

    if tasks['evaluate']:
        print('started evaluating')
        evl = []
        if JACCARD or CS_BENCH or PCA or PD:
            for method in config['methods']:
                if JACCARD or CS_BENCH:
                    cmd = (
                        f'pixi run -e eval'
                        f' python -m XenSegEval.eval.eval'
                        f' -c {config_path} -m {method}'
                    )
                    sbatch_kwargs['cmd'] = cmd
                    evl.append(
                        subprocess.Popen(
                            submit_sbatch(**sbatch_kwargs),
                            shell=True
                        )
                    )
                if (PCA and method in PCA_CAPABLE):
                    for section in sections:
                        cmd = (
                            f'pixi run -e free'
                            f' python -m XenSegEval.eval.free'
                            f' -c {config_path} -m {method} -s {section}'
                        )
                        sbatch_kwargs['cmd'] = cmd
                        evl.append(
                            subprocess.Popen(
                                submit_sbatch(**sbatch_kwargs),
                                shell=True
                            )
                        )
        if CROSS:
            cmd = (
                f'pixi run -e eval'
                f' python -m XenSegEval.eval.cross'
                f' -c {config_path}'
            )
            sbatch_kwargs['cmd'] = cmd
            evl.append(
                subprocess.Popen(
                    submit_sbatch(**sbatch_kwargs),
                    shell=True
                )
            )

        for p in evl:
            p.wait()

    print('done :3')
