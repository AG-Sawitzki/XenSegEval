from XenSegEval.utils import get_config_args, submit_sbatch

from pathlib import Path
import subprocess
import argparse
import os

import tomlkit


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

    sections = sections_dictionary.keys()

    for section in sections:
        spatialdata = f'{results}/{section}/spatialdata.zarr'
        all_plg = f'{results}/{section}/cell-polygons.geojson.gz'
        layer_plg = f'{results}/{section}/cell-polygons_layers.geojson.gz'
        counts = f'{results}/{section}/counts.mtx.gz'

        proseg_cmd = (
            'proseg '
            '--xenium '
            '--overwrite '
            '--output-spatialdata "{spatialdata}" '
            '--output-cell-polygons "{all_poylgons}" '
            '--output-cell-polygon-layers "{layer_polygons}" '
            '--output-counts "{counts}" '
            '"{processed}/{section}/transcripts/relative.csv.gz" '
        )
        subprocess.Popen(proseg_cmd, shell=True)
