from XenSegEval.plotting.utils import (
    heatmap,
    annotate_heatmap,
)

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    choices = ['all', 'f1', 'seg', 'jaccard', 'dice', 'PQ', 'F1', 'Jaccard']
    parser = argparse.ArgumentParser(prog='plot_cross')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    # parser.add_argument(
    #     '-b', '--Benchmark',
    #     default='',
    #     choices=['cs', 'u4n', ''],
    #     help=(
    #         'Which benchmark to use.'
    #         ' `u4n`, `cs` or none. If none do both.'
    #     )
    # )
    parser.add_argument(
        '-m', '--Metric',
        default='all',
        choices=choices,
        help=(
            'Which metric(1) to load and plot for.'
            ' lower case and PQ are from `cs`'
            ' upper case are from `u4n`'
            ' enter `all` to plot for all metrics'
        )
    )
    parser.add_argument(
        '-s', '--Section',
        help='Section to plot for.'
    )

    args = parser.parse_args()

    config_path = args.Config
    # benchmark = [args.Benchmark]
    metric = args.Metric
    section = args.Section

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'plot')
    globals().update(variables)

    cmap = ListedColormap([h for _, h in cmap.items()])

    if metric == 'all':
        metrics = choices[1:]
    else:
        metrics = [metric]

    for metric in metrics:
        if metric.istitle():
            benchmark = 'u4n'
        else:
            benchmark = 'cs'
        path = Path(
            f'{results}/{metric}_{benchmark}'
            f'_cross_evaluation_{section}'
        )
        arr_path = path + '.npy'
        if path.is_file():
            cross_res = np.load(arr_path, allow_pickle=True)
            with open(path, 'r') as f:
                labels = f.load()
                labels = labels.split(' ')

            fig, ax = plt.subplots()

            im, cbar = heatmap(
                cross_res, labels, labels, ax=ax,
                cmap=cmap, cbarlabel=CROSS['metric']
            )
            texts = annotate_heatmap(im, valfmt='{x:.1f}')
            fig.tight_layout()
            fig.savefig(
                f'{results}/{metric}_{benchmark}'
                f'_cross_evaluation_{section}.pdf'
            )
