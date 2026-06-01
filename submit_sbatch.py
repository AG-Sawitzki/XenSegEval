import os
import argparse
import tomllib
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Make and start a job on the BIH-HPC-Cluster.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--CMD', help='command.')
    parser.add_argument('-s', '--Section', help='Section.')
    parser.add_argument('-g', '--GPU', store=False, help='bool. To use a gpu or not.')
    parser.add_argument('-d', '--Double', store=False, help='bool. If higher mem & cpu should be use.')
    args = parser.parse_args()

    method = args.CMD
    section = args.Section
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

    if not GPU:
        job_kwargs['gpu'] = ""
    if Double:
        job_kwargs['time'] *= 2
        job_kwargs['mem'] *= 2
        job_kwargs['cpu'] *= 2

    log_path = Path(f"{home}/{sample}/run/{method}/{logs}/{name}")
    cmd = f"{Source or Python} {file} {args}"


    job_file = NamedTemporaryFile(
        mode="w+b",
        dir=tempfile_dir,
        delete=False,
        delete_on_close=False
    )

    with open(job_file) as fh:
        fh.writelines(
            """#!/bin/bash
            #
            #SBATCH --job-name={name}
            #SBATCH --gres={gpu}
            #SBATCH --time=2-00
            #SBATCH --mem={mem}
            #SBATCH --cpus-per-task={cpu}
            #SBATCH --output={output}/%N_%j.out
            #SBATCH --error={log_path}/%N_%j.err
            #SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50
            #SBATCH --mail-user={mail}
            #
            {cmd}""".format(**kwargs)
        )

        os.system(f"sbatch {job_file}")