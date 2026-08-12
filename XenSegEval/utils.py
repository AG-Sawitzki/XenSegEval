import os
import gzip
import json
import psutil
from time import sleep
from pathlib import Path

import tomlkit
import numpy as np

# types
from numpy.typing import ArrayLike
from typing import Any, Union
from pathlib import PosixPath


def depth(
    d: dict
) -> int:
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0



def get_memory_usage_percentage() -> float:
    """Get the memory usage as percentage.

    Returns
    ----------
        out : float
            Float of currently used memory in percentage.
    """
    process = psutil.Process()
    # Total system memory in bytes
    total_memory = psutil.virtual_memory().total
    # Resident Set Size in bytes
    mem_info = process.memory_info()
    used_memory = mem_info.rss
    # Calculate percentage
    memory_percentage = (used_memory / total_memory) * 100
    return memory_percentage


def get_section_coords(
    dictionary: Union[dict, str, os.PathLike[Any], PosixPath],
    key: str,
) -> tuple[int, int]:
    '''Get w,h from a dictionary organized as described in README.md

    Parameters
    ----------
        dictionary : dict or Path
            Dictionary or Path to a json-file containing the dictionary.
        key : str
            key/section name of coordinats.

    Returns
    ----------
        height and width of given rectangle.
    '''
    if type(dictionary) in [str, os.PathLike, PosixPath]:
            with open(dictionary) as file:
                dictionary = json.load(file)
    
    assert type(dictionary) is dict, \
        f'Input is wrong type: {type(dictionary)}'
    assert type(key) is str, f'key is not str: {type(key)}'

    coords = dictionary[key]
    x_coords = coords['x']
    y_coords = coords['y']

    return x_coords, y_coords


def get_section_dims(
    dictionary: Union[dict, str, os.PathLike[Any], PosixPath],
    key: str,
) -> tuple[int, int]:
    '''Get w,h from a dictionary organized as described in README.md

    Parameters
    ----------
        dictionary : dict or Path
            Dictionary or Path to a json-file containing the dictionary.
        key : str
            key/section name of coordinats.

    Returns
    ----------
        height and width of given rectangle.
    '''
    if type(dictionary) in [str, os.PathLike, PosixPath]:
        with open(dictionary) as file:
            dictionary = json.load(file)

    assert type(dictionary) is dict, \
        f'dictionary is wrong type: {type(dictionary)}'
    assert type(key) is str, f'key is not str: {type(key)}'

    x_coords, y_coords = get_section_coords(dictionary, key)

    width = x_coords[1] - x_coords[0]
    height = y_coords[1] - y_coords[0]

    return height, width


def submit_sbatch(
    job_dir: Union[str, os.PathLike[Any], PosixPath],
    time: int,
    mem: int,
    cpu: int,
    log_path: Union[str, os.PathLike[Any], PosixPath],
    cmd: str,
    gpu: Union[str, None] = None,
    mail: Union[str, None] = None,
) -> str:
    '''Writes a job-file for sbatch and returns the command to submit it.

    Parameters
    ----------
        tempfile_dir : Path
            Path to a directory
            Where the sbatch files will be saved.
        time : int
            Days to reserve the node for.
        mem : int
            How much RAM in GB to request.
        cpu :  int
            How many cpu-cores to request.
        log_path : Path
            Path to directory
            Where the logs will be saved.
        cmd : str
            The command to run on the node.
        gpu : str, optional
            Wether to run on a gpu node or not.
            Default is `None`.
        mail : str, optional
            The mail-address to send sbatch updates to.
            Default is `None`.

    Returns
    ----------
        out : str
            String with which the job can be submitted
    '''
    if cmd.partition(' ')[0] == 'bash':
        file = cmd[cmd.find('Xen'):cmd.rfind('.sh')]
        name = Path(file).stem
    else:
        file = cmd[cmd.find('Xen'):cmd.find(' -c')]
        name = file.rpartition('.')
        name = name[-1]
        name = name.replace('-', '')
        if name == 'eval':
            name += '_'+cmd[cmd.rfind('-m')+3:]
        if name == 'free':
            name += '_'+cmd[cmd.rfind('-m')+3:cmd.rfind('-s')-1]
    with open(f'{job_dir}/{name}.sh', 'w+') as fh:
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
            fh.writelines(
                f'#SBATCH --mail-type=BEGIN,END,FAIL,'
                'TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('\n#\n')
        fh.writelines(f'pixi run {cmd}\n')

    sleep(2)

    return f'sbatch {fh.name}'


def get_config_args(
    config: Union[str, os.PathLike[Any], PosixPath, dict],
    method: Union[str] = ''
) -> dict:
    '''Return config

    Parameters
    ----------
        config : Path or dict
            Path to config.toml or dict of parsed config.
        method : str, optional
            String of pipeline step.
            Default is `''` (empty).

    Returns
    ----------
        out : dict
            Dictionary of parsed config.
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
    plotting = config['plotting']

    if 'mail' in owner:
        mail = owner['mail']
    else:
        mail = None

    home = Path(paths['home'])
    data_path = Path(paths['data_path'])
    sample_name = paths['sample_name']
    gt_path = Path(paths['gt_path'])
    # define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = Path(paths['sections_path'])
    else:
        sections_path = processed / 'sections_px.json'

    with open(sections_path) as f:
        section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    # define processed and results directory
    processed = Path(f'{home}/{sample_name}/processed/')
    processed.mkdir(parents=True, exist_ok=True)
    results = Path(f'{home}/{sample_name}/results/')
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample_name}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    job_dir = Path(f'{home}/{sample_name}/run/jobs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    sbatch_kwargs.update(
        log_path=str(log_path),
        job_dir=str(job_dir),
        mail=mail,
    )

    variables.update(dict(
        home=home,
        data_path=data_path,
        sample_name=sample_name,
        sections_path=sections_path,
        sections=sections,
        processed=processed,
        results=results,
        sbatch_kwargs=sbatch_kwargs,
        imagestats=imagestats,
        section_dictionary=section_dictionary
    ))

    if gt_path:
        variables.update(dict(
            gt_path=gt_path,
        ))

    if method in methods:
        results = Path(results / f'{method}/output/')
        results.mkdir(parents=True, exist_ok=True)
        variables.update(dict(
            method=methods[method],
            results=results,
            pixelsizeXY=imagestats['pixelsize_xy'],
            planes=preprocessing['planes']
        ))
    else:
        if method == 'eval':
            variables.update(dict(
                methods=methods,
                PD=evaluation['pd'],
                PCA=evaluation['pca'],
                JACCARD=evaluation['jaccard'],
                CS_BENCH=evaluation['cs_bench'],
            ))
        elif method == 'cross':
            variables.update(dict(
                methods=methods,
                CROSS=evaluation['cross']
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
                tasks=config['tasks'],
                include_xenium=evaluation['include_xenium'],
                PD=evaluation['pd']['use'],
                PCA=evaluation['pca']['use'],
                JACCARD=evaluation['jaccard']['use'],
                CS_BENCH=evaluation['cs_bench']['use'],
                CROSS=evaluation['cross']['use'],
                CROSS_METRIC=evaluation['cross']['metric'],
                PLOT=plotting,
            ))
        elif method == 'plot':
            # colors = [
            #     methods[method]['color'] for method in methods
            # ]
            variables.update(dict(
                pixelsizeXY=imagestats['pixelsize_xy'],
                colors=plotting['colors'],
                cmap=plotting['cmap'],
                CROSS=evaluation['cross'],
                methods=methods,
            ))

    return variables
