
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    cross_eval,
    mean_cross_eval
)

import argparse
from pathlib import Path

import tomlkit
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CrossEval.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'cross')
    globals().update(variables)

    run = Path(f'{home}/{sample_name}/run/')

    if CROSS['use']:
        for section in sections:
            metric = CROSS['metric']
            CROSS_args = dict(
                results=results,
                methods=methods,
                section=section,
                metric=metric,
                threshold=CROSS['threshold']
            )
            if CROSS['include_gt']:
                CROSS_args.update({'gt_path': gt_path})
            res, labels = cross_eval(
                **CROSS_args
            )
            print(labels)
            print(res)

            # cross_with_avg, avg_hori, avg_vert = mean_cross_eval(
            #     res,
            #     methods,
            #     labels
            # )

            # print(cross_with_avg)

            path = (
                f'{results}/{CROSS["metric"]}_cross_evaluation_{section}'
            )
            np.save(path + '.npy', res)
            # np.save(path + '_avg.npy', cross_with_avg)
            with open(path, 'w') as f: 
                f.write(' '.join(labels))
