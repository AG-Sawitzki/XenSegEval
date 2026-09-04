from XenSegEval.utils import (
    get_config_args,
)
from XenSegEval.plotting.utils import (
    heatmap,
    annotate_heatmap,
)

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tomlkit

from matplotlib.colors import (
    ListedColormap,
    LinearSegmentedColormap,
)

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
            'Which metric to load and plot for.'
            ' lower case and PQ are from `dc`'
            ' upper case are from `u4n`'
            ' enter `all` to plot for all metrics available'
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

    print('starting')

    nodes = [0.0, 0.3, 0.4, 0.5, 0.75, 1.0]
    cmap = LinearSegmentedColormap.from_list(
        'charite',
        list(zip(nodes, [h for _, h in cmap.items()][::-1]))
    )
    if metric == 'all':
        metrics = choices[1:]
    else:
        metrics = [metric]


    print(metrics)
    for metric in metrics:
        if metric.istitle():
            benchmark = 'u4n'
            path = f'{results}/{metric}_u4n_cross_evaluation_{section}'
            label_path = path + '_labels'
            arr_path = path + '.npy'
        else:
            benchmark = 'dc'
            arr_path = f'{results}/{section}_{metric}_CROSS.npy'
            label_path = f'{results}/{section}_DC-TOOLS_labels'

        print(benchmark)
        if Path(arr_path).is_file():
            cross_res = np.load(arr_path, allow_pickle=True)
            with open(label_path, 'r') as f:
                labels = f.read()
                labels = labels.split(' ')
            # labels.append('avg')
            fig, ax = plt.subplots()

            im, cbar = heatmap(
                cross_res, labels, labels, ax=ax,
                cmap=cmap, cbarlabel=metric
            )
            texts = annotate_heatmap(im, valfmt='{x:.1f}')
            fig.tight_layout()
            fig.savefig(
                f'{results}/{metric}_{benchmark}'
                f'_cross_evaluation_{section}.pdf'
            )
