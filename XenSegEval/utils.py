import os
import argparse
import tomlkit
import pickle
from pathlib import Path

# types
from typing import Any

def submit_sbatch(
    job_kwargs: dict,
    gpu: bool = False
) -> str:
    '''Writes a job-file for sbatch akd retruns the command to submit it.
    Args:
        job_kwargs: a dictionary containing the variables used for formatting. --should be split up and then used via **job_kwargr--
        gpu: wether to run on a gpu node or not.
    Returns:
        string with which the job can be submitted
    '''
    cmd = job_kwargs['cmd']
    name = cmd[cmd.find('/')+1:cmd.find('.')]
    with open(f'{tempfile_dir}/{name}.sh'.format(**job_kwargs), 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines('#SBATCH --job-name={name}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --wait')
        if gpu:
            fh.writelines('#SBATCH --gres={gpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --time={time}-00\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mem={mem}G\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --cpus-per-task={cpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --output={log_path}/%N_%j.out\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --error={log_path}/%N_%j.err\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#SBATCH --mail-user={mail}\n'.format(**job_kwargs))
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('#\n')
        fh.writelines('pixi run {cmd}\n'.format(**job_kwargs))

    return f'sbatch {fh.name}'


def submit_cmd(cmd, config='config.toml', gpu=False, double=False):
    '''DO NOT USE.
    '''
    submit_str = f'python submit_sbatch.py -c {config} -m "{cmd}"'
    #if section is not None:
    #    submit_str += f' -s {section}'
    if gpu:
        submit_str += ' -g'
    if double:
        submit_str += ' -d'
    return submit_str


def get_config_args(
    config: str | os.PathLike[Any] | dict,
    method: str | None = None
) -> dict:
    '''Return config
    Args:
        config: string or path to config.toml or dict of parsed config.
        method(optional): string of segmentation method.
    '''
    tasks = config['Tasks']
    paths = config['paths']
    imagestats = config['ImageStats']
    sbatch_kwargs = config['sbatch']
    preprocessing = config['preprocessing']
    methods = config['methods']
    evaluation = config['evaluation']

    if method is not None:
        method=methods[method]
    else:
        method = None

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
        mail=config['owner']['mail'],
    )

    args = dict(
        tasks=tasks,
        home=home,
        data_path=data_path,
        sample_name=sample_name,
        gt_path=gt_path,
        sections_path=sections_path,
        processed=processed,
        results=resutls,
        log_path=log_path,
        sbatch_kwargs=sbatch_kwargs,
        method=method,
        PD = evaluation['PD'],
        PCA = evaluation['PCA'],
        JACCARD = evaluation['JACCARD'],
        CS_BENCH = evaluation['CS-BENCH'],
    )

    return args