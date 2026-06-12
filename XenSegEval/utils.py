import os
import argparse
import tomlkit
import pickle
from pathlib import Path

# types
from typing import Any

def submit_sbatch(
    tempfile_dir: str | os.PathLike,
    time: str | int,
    log_path: str | os.PathLike,
    cmd: str,
    gpu: str | None = None,
    mail: str | None = None,
) -> str:
    '''Writes a job-file for sbatch akd retruns the command to submit it.
    Args:
        tempfile_dir: path to a directory 
                      where the sbatch files will be saved.
        time: days to reserve the node for.
        log_path: path to directory
                  where the logs will be saved.
        cmd: the command to run on the node.
        gpu(optional): wether to run on a gpu node or not.
        mail(optional): the mail-address to send sbatch updates to.
    Returns:
        string with which the job can be submitted
    '''
    cmd = sbatch_kwargs['cmd']
    name = cmd[cmd.find('/')+1:cmd.find('.')]
    with open(f'{tempfile_dir}/{name}.sh', 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines(f'#SBATCH --job-name={name}\n')
        fh.writelines('#SBATCH --wait')
        if gpu is not None:
            fh.writelines(f'#SBATCH --gres={gpu}\n')
        fh.writelines(f'#SBATCH --time={time}-00\n')
        fh.writelines(f'#SBATCH --mem={mem}G\n')
        fh.writelines(f'#SBATCH --cpus-per-task={cpu}\n')
        fh.writelines(f'#SBATCH --output={log_path}/%N_%j.out\n')
        fh.writelines(f'#SBATCH --error={log_path}/%N_%j.err\n')
        if mail is not None:
            fh.writelines(f'#SBATCH --mail-user={mail}\n')
            fh.writelines(f'#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('#\n')
        fh.writelines(f'pixi run {cmd}\n')

    return f'sbatch {fh.name}'


def get_config_args(
    config: str | os.PathLike[Any] | dict,
    method: str | None = None
) -> dict:
    '''Return config
    Args:
        config: string or path to config.toml or dict of parsed config.
        method: string of pipeline step.
    Returns:
        dictionary of parsed config.
    '''
    variables = dict()

    if type(config) is not dict:
            with open(config, 'rb') as f:
                config = tomlkit.load(f)

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

    home = paths['home']
    data_path = paths['data_path']
    sample_name = paths['sample_name']
    gt_path = paths['gt_path']
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = paths['sections_path']
    else:
        sections_path = processed / 'sections_px.json'
    
    ## define processed and results directory
    processed = Path(f'{home}/{sample}/processed/')
    processed.mkdir(parents=True, exist_ok=True)
    results = Path(f'{home}/{sample}/results/')
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)

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
        results=resutls,
        sbatch_kwargs=sbatch_kwargs,
    ))

    if method is in methods:
        variables.update(dict(
            method=methods[method],
            pixelsizeXY = imagestats['pixelsize_xy']
        ))
    else:
        if method == 'eval':
            variables.update(dict(
                gt_path=paths['gt_path'],
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
                pixelsizeXY=imagestats['pixelsize_xy'],
                pixelsizeZ=imagestats['pixelsize_z'],
            ))
        elif method == 'main':
            variables.update(dict(
                tasks=config['Tasks']
            ))

    return variables