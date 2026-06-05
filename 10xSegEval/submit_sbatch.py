import os
import argparse
import tomllib
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile


def make_cmd(cmd, config='config.toml', gpu=False, double=False):
    submit_str = f'python submit_sbatch.py -c {config} -m "{cmd}"'
    if gpu:
        submit_str += ' -g'
    if double:
        submit_str += ' -d'

    return submit_str


def submit_sbatch(job_kwargs, gpu=False, double=False):
    cmd = job_kwargs['cmd']
    name = cmd[cmd.find('/')+1:cmd.find('.')]

    if not gpu:
        job_kwargs['gpu'] = ''
    if double:
        job_kwargs['time'] = job_kwargs['time']*2
        job_kwargs['mem'] = job_kwargs['mem']*2
        job_kwargs['cpu'] = job_kwargs['cpu']*2
    with open(f'{tempfile_dir}/{name}.job'.format(**job_kwargs), 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines('#SBATCH --job-name={name}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --wait')
        fh.writelines('#SBATCH --gres={gpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --time={time}-00\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mem={mem}G\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --cpus-per-task={cpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --output={log_path}/%N_%j.out\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --error={log_path}/%N_%j.err\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#SBATCH --mail-user={mail}\n'.format(**job_kwargs))
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        # fh.writelines('micromamba activate simple\n')
        fh.writelines('pixi shell segeval')
        fh.writelines('#\n')
        fh.writelines('{cmd}\n'.format(**job_kwargs))

    #os.system(f'sbatch {fh.name}')

    return f'sbatch {fh.name}'


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
        job_kwargs['time'] = job_kwargs['time']*2
        job_kwargs['mem'] = job_kwargs['mem']*2
        job_kwargs['cpu'] = job_kwargs['cpu']*2

    name = cmd[cmd.find('/')+1:cmd.find('.')]
    if name in config['methods']:
        log_path = Path(f'{home}/{sample}/run/methods/{name}/logs/')
    else:
        log_path = Path(f'{home}/{sample}/run/processing/{name}/logs/')

    log_path.mkdir(parents=True, exist_ok=True)

    job_kwargs.update(
        dict(
            cmd=cmd,
            log_path=log_path,
            mail=config['owner']['mail'],
            name=name
        )
    )
    tempfile_dir = Path(f'{os.getcwd()}/temp/')
    tempfile_dir.mkdir(parents=True, exist_ok=True)

    with open(f'{tempfile_dir}/{name}.job', 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines('#SBATCH --job-name={name}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --wait')
        fh.writelines('#SBATCH --gres={gpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --time={time}-00\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mem={mem}G\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --cpus-per-task={cpu}\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --output={log_path}/%N_%j.out\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --error={log_path}/%N_%j.err\n'.format(**job_kwargs))
        fh.writelines('#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#SBATCH --mail-user={mail}\n'.format(**job_kwargs))
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('micromamba activate simple\n')
        fh.writelines('#\n')
        fh.writelines('{cmd}\n'.format(**job_kwargs))

    os.system(f'sbatch {fh.name}')