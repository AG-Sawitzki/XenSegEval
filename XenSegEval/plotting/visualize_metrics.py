from XenSegEval.utils import get_config_args
from XenSegEval.plotting.utils import (
    # bar_method_eval,
    bar_compare_eval
)

import argparse
import tomlkit
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Metrics')
    parser.add_argument(
        '-b', '--Benchmark',
        default=None,
        help=(
            'Which evaluation methods metrics to plot.'
            ' `u4n`, `cs` or none.If none do both.'
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

    benchmarks = ['cs', 'u4n']

    fig, ax = plt.subplots()

    if benchmark in benchmarks:
            bar_compare_eval(
                methods, results, section,
                fig, ax, colors,
                benchmark
            )
    elif not benchmark:
        for benchmark in benchmarks:
            bar_compare_eval(
                methods, results, section,
                fig, ax, colors,
                benchmark
            )
    else:
        print(f'Given benchmark not available. Choose from {benchmarks}')