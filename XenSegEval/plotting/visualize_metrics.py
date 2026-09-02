from XenSegEval.utils import get_config_args
from XenSegEval.plotting.utils import (
    # bar_method_eval,
    bar_compare_eval
)

import argparse
import tomlkit
import matplotlib.pyplot as plt

if __name__ == '__main__':
    choices = ['all', 'u4n', 'dc', 'area']
    parser = argparse.ArgumentParser(prog='Metrics')
    parser.add_argument(
        '-b', '--Benchmark',
        default=None,
        choices=choices,
        help=(
            'Which evaluation methods metrics to plot.'
            ' `u4n`, `dc` or none. If none do both.'
        )
    )
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-s', '--Section',
        help='Section to plot for.'
    )

    args = parser.parse_args()

    config_path = args.Config
    benchmark = args.Benchmark
    section = args.Section

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'plot')
    globals().update(variables)

    if benchmark in choices:
        if benchmark == 'all':
            benchmarks = choices[1:]
        else:
            benchmarks = [benchmark]
        for benchmark in benchmarks:
            fig, ax = plt.subplots()
            bar_compare_eval(
                methods, results, section,
                fig, ax, colors,
                benchmark,
            )
    else:
        print(f'Given benchmark not available. Choose from {choices}')
