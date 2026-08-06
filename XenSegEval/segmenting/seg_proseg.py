from XenSegEval.utils import get_config_args, submit_sbatch

from pathlib import Path
import subprocess
import argparse
import os

import tomlkit

from typing import Any, Union
from pathlib import PosixPath


def add_arg(
    arg_list: list,
    arg: str,
    value: Union[str, int, bool],
) -> list:
    if isinstance(value, bool):
        if value:
            arg_list.append(f'--{arg}')
    else:
        arg_list.append(f'--{arg} {value}')

    return arg_list


def get_arguments(
    results: Union[str, os.PathLike, PosixPath],
    section: Union[str, int],
    method: dict,
    planes: int,
    pixelsizeXY: float
) -> str:
    '''Get the arguments for proseg from the config file. Constructs a command.

    Parameters
    ----------
        results
            Path to the results folder.
        section
            Name of the section the segmentation will run on.
        method
            The parsed config table for proseg. 
            Or a dictionary of all arguments in its config table.
        planes
            Amount of z-Layers included when processing the images.
            Based on `preprocessing.planes`.
        pixelsizeXY
            The size of a pixel in XY dimensions. In microns.
            Based on `ImageStats.pixelsize_xy`

    Returns
    ----------
        out
            A command as string to execute proseg with.
    '''
    base = 'proseg --xenium'

    outputs = []
    for arg, value in method['output'].items():
        arg = 'output-' + arg
        if 'path' in arg:
            if value:
                path = value
            else:
                path = f'{results}/{section}/'
            outputs = add_arg(outputs, arg, path)
        else:
            if value:
                if 'counts' in arg:
                    value = f'{arg}.mtx.gz'
                elif 'meta' in arg or 'rates' in arg:
                    value = f'{arg}.csv.gz'
                elif 'polygon' in arg:
                    value = f'{arg}.geojson.gz'
                outputs = add_arg(outputs, arg, value)

    general = []
    for arg, value in method['general'].items():
        if not value:
            if arg == 'nthreads':
                general = add_arg(general, arg, value)
            if arg == 'output-spatialdata':
                value = f'{path}/spatialdata.zarr'
                general = add_arg(general, arg, value)
        else:
            if arg == 'voxel-layers':
                if value == 'planes':
                    general = add_arg(general, arg, planes)
                else:
                    general = add_arg(general, arg, value)
            elif arg == 'overwrite':
                general = add_arg(general, arg, value)
            elif arg == 'voxel-size':
                general = add_arg(general, arg, pixelsizeXY)
            elif arg == 'burnin-voxel-size':
                general = add_arg(general, arg, pixelsizeXY*8)
            else:
                general = add_arg(general, arg, value)

    model = []
    for arg, value in method['model'].items():
        model = add_arg(model, arg, value)

    diffusion = []
    if not method['diffusion']['no-diffusion']:
        for arg, value in method['diffusion'].items():
            diffusion = add_arg(diffusion, arg, value)

    arguments = outputs + general + model + diffusion

    print(arguments)

    cmd = ' '.join(arguments)
    cmd = ' '.join((base, cmd))

    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ProSeg')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Optional. Path to a config file like "config.toml".'
    )
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'proseg')
    globals().update(variables)

    for section in sections:
        cmd = get_arguments(
            results=results,
            section=section,
            method=method,
            planes=len(planes),
            pixelsizeXY=pixelsizeXY
        )

        SRT_path = f'{processed}/{section}/transcripts/relative.csv.gz'

        proseg_cmd = ' '.join((cmd, SRT_path))

        print(proseg_cmd)
        subprocess.Popen(proseg_cmd, shell=True)
