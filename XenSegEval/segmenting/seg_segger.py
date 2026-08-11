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


def get_arguments_segger(
    # results: Union[str, os.PathLike, PosixPath],
    # section: Union[str, int],
    method: dict,
    # planes: int,
    # pixelsizeXY: float
) -> str:
    '''Get the arguments for proseg from the config file. Constructs a command.

    Parameters
    ----------
        # results
            # Path to the results folder.
        # section
            # Name of the section the segmentation will run on.
        method
            The parsed config table for proseg. 
            Or a dictionary of all arguments in its config table.
        # planes
            Amount of z-Layers included when processing the images.
            Based on `preprocessing.planes`.
        # pixelsizeXY
            # The size of a pixel in XY dimensions. In microns.
            # Based on `ImageStats.pixelsize_xy`

    Returns
    ----------
        out
            A command as string to execute segger.
    '''
    base = 'segger segment'

    in_out = add_arg([], 'save-anndata', method['save-anndata'])

    node = []
    for arg, value in method['node'].items():
        if not value:
            continue
        else:
            node = add_arg(node, arg, value)

    transcripts = []
    for arg, value in method['transcripts'].items():
        transcripts = add_arg(transcripts, arg, value)

    prediction = []
    for arg, value in method['prediction'].items():
        if 'mode' in arg:
            continue
        else:
            prediction = add_arg(prediction, arg, value)

    tiling = []
    for arg, value in method['tiling'].items():
        tiling = add_arg(tiling, arg, value)

    model = []
    for arg, value in method['model'].items():
        model = add_arg(model, arg, value)

    loss = []
    for arg,value in method['loss'].items():
        loss = add_arg(loss, arg, value)

    arguments = in_out + node + transcripts + prediction + tiling + model + loss

    segment_cmd = ' '.join(arguments)
    segment_cmd = ' '.join((base, segment_cmd))

    
    export = []
    for arg, value in method['export'].items():
        if 'output' in arg:
            output = value
        else:
            export = add_arg(export, arg, value)

    export_output = ' '.join(output)
    export_args = ' '.join(export)

    export_cmd = f'segger export {export_output}'
    export_cmd = ' '.join((export_cmd, export_args))

    return segment_cmd, export_cmd


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

    variables = get_config_args(config, 'segger')
    globals().update(variables)

    for mode in method['prediction']['prediction-mode']:
        segment_cmd, export_cmd = get_arguments_segger(
            method=method,
        )
        prediction_mode = f'--prediction-mode {mode}'
        segment_cmd = ' '.join((segment_cmd, prediction_mode))

        out = Path(f'{results}/{mode}/')
        out.mkdir(parents=True, exist_ok=True)

        input_output = f'-i {data_path} -o {out}'
        segment_cmd = ' '.join((segment_cmd, input_output))

        print(f'Predicting {mode}.')
        # seggerT = subprocess.Popen(segment_cmd, shell=True)
        # seggerT.wait()

        input_output = ' '.join((f' -s {out}/segger_segmentation.parquet', input_output))
        export_cmd = ' '.join((export_cmd, input_output))
        print(f'Exporting for {mode}.')
        # exportT = subprocess.Popen(export_cmd, shell=True)
        # exportT.wait()

        print(f'Preparing GeoJSON for {mode}.')
        prepare_cmd = f'python -m XenSegEval.processing.prepare_segger -m {mode}'
        prepareT = subprocess.Popen(prepare_cmd, shell=True)
        prepareT.wait()