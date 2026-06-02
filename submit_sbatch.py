import os
import argparse
import tomllib
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='SUBMIT',
        description='Make and start a job on the BIH-HPC-Cluster.'
    )
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    parser.add_argument('-m', '--CMD', help='Command to run on cluster.')
    parser.add_argument(
        '-g', '--GPU',
        default=False,
        action='store_true',
        help='bool. To use a gpu or not.'
    )
    parser.add_argument(
        '-d', '--Double',
        default=False,
        action='store_true',
        help='bool. If higher mem & cpu should be use.'
    )
    args = parser.parse_args()

    cmd = args.CMD
    config_path = args.Config
    double = args.Double
    gpu = args.GPU

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    paths = config['paths']

    # define paths
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']
    job_kwargs = config['job']

    if not gpu:
        job_kwargs['gpu'] = ''
    if double:
        job_kwargs['time'] *= 2
        job_kwargs['mem'] *= 2
        job_kwargs['cpu'] *= 2

    log_path = Path(f"{home}/{sample}/run/{cmd}/logs/")

    job_kwargs.update(
        dict(
            cmd=cmd,
            log_path=log_path,
            mail=config['owner']['mail']
        )
    )
    tempfile_dir = Path(f'{os.getcwd()}/temp/')
    tempfile_dir.mkdir(parents=True, exist_ok=True)

    with open(f'{tempfile_dir}/test.job', 'w') as fh:
        fh.writelines(
            '''#!/bin/bash
            #
            #SBATCH --job-name=test
            #SBATCH --gres={gpu}
            #SBATCH --time={time}-00
            #SBATCH --mem={mem}G
            #SBATCH --cpus-per-task={cpu}
            #SBATCH --output={log_path}/%N_%j.out
            #SBATCH --error={log_path}/%N_%j.err
            #SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50
            #SBATCH --mail-user={mail}
            #
            {cmd}'''.format(**job_kwargs)
        )

        os.system(f'sbatch {tempfile_dir}{fh.name}')