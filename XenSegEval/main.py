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
    # 'stardist',
    'segger',
    'xenium'
]
'''LIST OF CURRENTLY SUPPORTED ALGORITHMS FOR CSEs PCA analysis'''

TO_MASK = [
    'proseg',
    'segger',
    'xenium'
]
'''LIST OF ALGORITHMS THAT ONLY PROVIDE POLYGONS AS OUTPUT.'''


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

    variables = get_config_args(config, 'main')
    globals().update(variables)

    methods = list(config['methods'])

    if (
        (sections is None or gt_path is None)
        and tasks['evaluate'] is True
    ):
        print(
            'No section and/or ground truth provided for evaluation.'
            ' Please check "[preprocessing]".'
        )
    if (
        (gt_path and not gt_name)
        or ((gt_path and gt_name) and (gt_name not in sections))
    ):
        print(
            'No name given to the GroundTruth. See config `paths.gt_name`.'
            '\n Or the given name is not in the provided dictionary.'
            'See `paths.sections_path`'
        )

    gpu = sbatch_kwargs['gpu']
    mem = sbatch_kwargs['mem']
    del sbatch_kwargs['gpu']
    # preprocess
    if tasks['preprocess']:
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

        if 'xenium' in methods:
            cmd = (
                'pixi run python -m XenSegEval.processing.prepare_xenium'
                f' -c {config_path}'
            )
            sbatch_kwargs['cmd'] = cmd
            print('Preparing Xenium Boundaries.')
            pX = subprocess.Popen(submit_sbatch(**sbatch_kwargs), shell=True)
            
            pX.wait()
        pI.wait()
        pT.wait()

    variables = get_config_args(config, 'main')
    globals().update(variables)

    if tasks['segment']:
        sbatch_kwargs['gpu'] = gpu
        print('started segmenting')
        seg = []
        for method in methods:
            cmd = f'bash XenSegEval/start/{method}.sh {config_path}'
            if method == 'segger':
                cmd = f'bash XenSegEval/start/{method}.sh {config_path}'
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
        for p in seg:
            p.wait()
        del sbatch_kwargs['gpu']


    if not tasks['skip_prepare']:
        print(f'Preparing masks and polygons.')
        preparing = []
        for method in methods:
            if method in TO_MASK:
                cmd = f'pixi run python -m XenSegEval.processing.polygon_to_mask -m {method}'
                sbatch_kwargs['cmd'] = cmd
                preparing.append(
                    subprocess.Popen(
                        submit_sbatch(**sbatch_kwargs),
                        shell=True
                    )
                )
            else:
                cmd = ('pixi run python -m XenSegEval.processing.mask_to_polygon'
                    f' -m {method}')
                sbatch_kwargs['cmd'] = cmd
                preparing.append(
                    subprocess.Popen(
                        submit_sbatch(**sbatch_kwargs),
                        shell=True
                    )
                )
        for p in preparing:
            p.wait()


    # methods.append('xenium')
    if tasks['evaluate']:
        print('started evaluating')
        evl = []
        if JACCARD or DC_TOOLS or PCA:
            for method in methods:
                if JACCARD or DC_TOOLS:
                    cmd = (
                        f'pixi run -e eval'
                        f' python -m XenSegEval.eval.masked.eval'
                        f' -c {config_path} -m {method} -gts {gt_name}'
                    )
                    sbatch_kwargs['cmd'] = cmd
                    evl.append(
                        subprocess.Popen(
                            submit_sbatch(**sbatch_kwargs),
                            shell=True
                        )
                    )
                if (PCA and method in PCA_CAPABLE):
                    sbatch_kwargs['gpu'] = gpu
                    for section in sections:
                        cmd = (
                            f'pixi run -e free'
                            f' python -m XenSegEval.eval.free.free'
                            f' -c {config_path} -m {method} -s {section}'
                        )
                        sbatch_kwargs['cmd'] = cmd
                        evl.append(
                            subprocess.Popen(
                                submit_sbatch(**sbatch_kwargs),
                                shell=True
                            )
                        )
                    del sbatch_kwargs['gpu']
        if CROSS:
            cmd = (
                f'pixi run -e eval'
                f' python -m XenSegEval.eval.cross.cross'
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

    if tasks['plot']:
        cmds = []
        plots = []
        for section in sections:
            if PLOT['cross']:
                cmds.append(
                    f'pixi run python -m XenSegEval.plotting.visualize_cross_eval'
                    f' -c {config_path} -m {CROSS_METRIC} -s {section}'
                )
            if PLOT['bars']:
                cmds.append(
                    f'pixi run python -m XenSegEval.plotting.visualize_metrics'
                    f' -s {section} -b both'
                )
            if PLOT['overlay']:
                for method in methods:
                    cmds.append(
                        f'pixi run python -m XenSegEval.plotting.visualize_segmentation'
                        f' -c {config_path} -m {method} -s {section}'
                    )
        for cmd in cmds:
            plots.append(
                subprocess.Popen(
                    cmd, shell=True
                )
            )
        for p in plots:
            p.wait()

    print('done :3')
