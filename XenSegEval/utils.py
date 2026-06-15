import os
import argparse
import tomlkit
import pickle
from pathlib import Path

# types
from typing import Any, Union

def submit_sbatch(
    tempfile_dir: Union[str, os.PathLike[Any]],
    time: int,
    mem: int,
    cpu: int,
    log_path: Union[str, os.PathLike[Any]],
    cmd: str,
    gpu: Union[str, None] = None,
    mail: Union[str, None] = None,
) -> str:
    '''Writes a job-file for sbatch akd retruns the command to submit it.
    Args:
        tempfile_dir: path to a directory 
                      where the sbatch files will be saved.
        time: days to reserve the node for.
        mem: how much RAM to request.
        cpu: how many cpu-cores to request.
        log_path: path to directory
                  where the logs will be saved.
        cmd: the command to run on the node.
        gpu(optional): wether to run on a gpu node or not.
        mail(optional): the mail-address to send sbatch updates to.
    Returns:
        string with which the job can be submitted
    '''
    if cmd.partition(' ')[0] == 'bash':
        file = cmd[cmd.find('Xen'):cmd.rfind('.sh')]
        name = Path(file).stem
    else:
        file = cmd[cmd.find('Xen'):cmd.find(' -c')]
        name = file.rpartition('.')
        name = name[-1]
        name.replace('-','')
        if name == 'eval':
            name += '_'+cmd[cmd.rfind('-m')+3:]
    with open(f'{tempfile_dir}/{name}.sh', 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines(f'#SBATCH --job-name={name}\n')
        fh.writelines('#SBATCH --wait\n')
        if gpu is not None:
            fh.writelines(f'#SBATCH --gres={gpu}\n')
        fh.writelines(f'#SBATCH --time={time}-00\n')
        fh.writelines(f'#SBATCH --mem={mem}G\n')
        fh.writelines(f'#SBATCH --cpus-per-task={cpu}\n')
        fh.writelines(f'#SBATCH --output={log_path}/{name}_%N_%j.out\n')
        fh.writelines(f'#SBATCH --error={log_path}/{name}_%N_%j.err\n')
        if mail is not None:
            fh.writelines(f'#SBATCH --mail-user={mail}\n')
            fh.writelines(f'#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('\n#\n')
        fh.writelines(f'pixi run {cmd}\n')

    return f'sbatch {fh.name}'


def get_config_args(
    config: Union[str, os.PathLike[Any], dict],
    method: Union[str, None] = None
) -> dict:
    '''Return config
    Args:
        config: string or path to config.toml or dict of parsed config.
        method: string of pipeline step.
    Returns:
        dictionary of parsed config.
    '''
    variables = dict()

    if type(config) is str or type(config) is os.PathLike:
            with open(config, 'rb') as f:
                config = tomlkit.load(f)

    config = dict(config)

    owner = config['owner']
    paths = config['paths']
    imagestats = config['ImageStats']
    sbatch_kwargs = config['sbatch']
    preprocessing = config['preprocessing']
    methods = config['methods']
    evaluation = config['evaluation']

    if 'mail' in owner:
        mail = owner['mail']
    else:
        mail = None

    home = Path(paths['home'])
    data_path = Path(paths['data_path'])
    sample_name = paths['sample_name']
    gt_path = Path(paths['gt_path'])
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = Path(paths['sections_path'])
    else:
        sections_path = processed / 'sections_px.json'
    
    ## define processed and results directory
    processed = Path(f'{home}/{sample_name}/processed/')
    processed.mkdir(parents=True, exist_ok=True)
    results = Path(f'{home}/{sample_name}/results/')
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample_name}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    tempfile_dir = Path(f'{home}/{sample_name}/run/jobs/')
    tempfile_dir.mkdir(parents=True, exist_ok=True)

    sbatch_kwargs.update(
        log_path=str(log_path),
        tempfile_dir=str(tempfile_dir),
        mail=mail,
    )

    variables.update(dict(
        home=home,
        data_path=data_path,
        sample_name=sample_name,
        sections_path=sections_path,
        processed=processed,
        results=results,
        sbatch_kwargs=sbatch_kwargs,
        imagestats=imagestats
    ))

    if method in methods:
        results = Path(results / f'{method}/output/')
        results.mkdir(parents=True, exist_ok=True)
        variables.update(dict(
            method=methods[method],
            results=results,
            pixelsizeXY = imagestats['pixelsize_xy']
        ))
    else:
        if method == 'eval':
            variables.update(dict(
                gt_path=gt_path,
                PD=evaluation['PD'],
                PCA=evaluation['PCA'],
                JACCARD=evaluation['JACCARD'],
                CS_BENCH=evaluation['CS-BENCH'],
            ))
        elif method in [
            'transcripts',
            'images',
            'boundaries'
        ]:
            variables.update(dict(
                preprocessing=preprocessing,
                pixelsizeXY=imagestats['pixelsize_xy'],
                pixelsizeZ=imagestats['pixelsize_z'],
            ))
        elif method == 'main':
            variables.update(dict(
                tasks=config['Tasks']
            ))

    return variables