import os
import sys
import argparse

import dissect
from pathlib import Path

from tomlkit import load
import numpy as np

from XenSegEval.utils import get_config_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DISSECT.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = load(f)

    variables = get_config_args(config, 'dissect')
    globals().update(variables)

    # load sections
    sections = section_dictionary.keys()

    for section in sections:
        img_path = Path(
            processed /
            f'{section}/morphology/focus/focus.ome.tif'
        )
        gene_mtx_filename = Path(
            processed /
            f'{section}/transcripts/relative.csv'
        )
        config_file = Path(
            './XenSegEval/segmenting/dissect/dissect_config.yaml'
        )
        weights_file = Path(
            './XenSegEval/segmenting/dissect/dissect_weights.pth'
        )
        output_dir = Path(results / f'{section}')
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = dissect.segmentation(
            img_path=str(img_path),
            platform='xenium',
            gene_mtx_filename=str(gene_mtx_filename),
            config_file=str(config_file),
            weights_file=str(weights_file),
            output=str(output_dir)
        )
