import os
import gzip
import json
from time import sleep
from pathlib import Path
from multiprocessing import cpu_count, Pool

import tomlkit
import numpy as np
import cv2
from shapely import Polygon
import geopandas as gpd

# types
from numpy.typing import ArrayLike
from typing import Any, Union
from pathlib import PosixPath


def get_section_dims(
    dictionary: Union[dict, str, os.PathLike[Any], PosixPath],
    key: str,
) -> tuple[int, int]:
    '''Get w,h from a dictionary organized as described in README.md

    Parameters
    ----------
        dictionary : dict or Path
            Dictionary or Path to a json-file containing the dictionary.
        key : str
            key/section name of coordinats.

    Returns
    ----------
        height and width of given rectangle.
    '''
    if type(dictionary) in [str, os.PathLike, PosixPath]:
        with open(dictionary) as file:
            dictionary = json.load(file)

    assert type(dictionary) is dict, (
        'dictionary is wrong type:'
        f'{type(dictionary)}'
    )

    assert type(key) is str, f'key is not str: {type(key)}'
    coords = np.array(dictionary[key])
    height = coords[1][0] - coords[0][0]
    width = coords[1][1] - coords[0][1]

    return height, width


def submit_sbatch(
    job_dir: Union[str, os.PathLike[Any], PosixPath],
    time: int,
    mem: int,
    cpu: int,
    log_path: Union[str, os.PathLike[Any], PosixPath],
    cmd: str,
    gpu: Union[str, None] = None,
    mail: Union[str, None] = None,
) -> str:
    '''Writes a job-file for sbatch and returns the command to submit it.

    Parameters
    ----------
        tempfile_dir : Path
            Path to a directory
            Where the sbatch files will be saved.
        time : int
            Days to reserve the node for.
        mem : int
            How much RAM in GB to request.
        cpu :  int
            How many cpu-cores to request.
        log_path : Path
            Path to directory
            Where the logs will be saved.
        cmd : str
            The command to run on the node.
        gpu : str, optional
            Wether to run on a gpu node or not.
            Default is `None`.
        mail : str, optional
            The mail-address to send sbatch updates to.
            Default is `None`.

    Returns
    ----------
        out : str
            String with which the job can be submitted
    '''
    if cmd.partition(' ')[0] == 'bash':
        file = cmd[cmd.find('Xen'):cmd.rfind('.sh')]
        name = Path(file).stem
    else:
        file = cmd[cmd.find('Xen'):cmd.find(' -c')]
        name = file.rpartition('.')
        name = name[-1]
        name = name.replace('-', '')
        if name == 'eval':
            name += '_'+cmd[cmd.rfind('-m')+3:]
        if name == 'free':
            name += '_'+cmd[cmd.rfind('-m')+3:cmd.rfind('-s')-1]
    with open(f'{job_dir}/{name}.sh', 'w+') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#\n')
        fh.writelines(f'#SBATCH --job-name={name}\n')
        fh.writelines('#SBATCH --wait\n')
        if gpu is not None:
            fh.writelines(f'#SBATCH --gres={gpu}\n')
        fh.writelines(f'#SBATCH --time={time}-00\n')
        fh.writelines(f'#SBATCH --mem={mem}G\n')
        fh.writelines(f'#SBATCH --cpus-per-task={cpu}\n')
        fh.writelines(f'#SBATCH --output={log_path}/{name}_%N_%j.out\n')
        fh.writelines(f'#SBATCH --error={log_path}/{name}_%N_%j.err\n')
        if mail is not None:
            fh.writelines(f'#SBATCH --mail-user={mail}\n')
            fh.writelines(
                f'#SBATCH --mail-type=BEGIN,END,FAIL,'
                'TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('\n#\n')
        fh.writelines(f'pixi run {cmd}\n')

    sleep(2)

    return f'sbatch {fh.name}'


def get_config_args(
    config: Union[str, os.PathLike[Any], PosixPath, dict],
    method: Union[str] = ''
) -> dict:
    '''Return config

    Parameters
    ----------
        config : Path or dict
            Path to config.toml or dict of parsed config.
        method : str, optional
            String of pipeline step.
            Default is `''` (empty).

    Returns
    ----------
        out : dict
            Dictionary of parsed config.
    '''
    variables = dict()

    if type(config) is str or type(config) is os.PathLike:
        with open(config, 'rb') as f:
            config = tomlkit.load(f)

    config = dict(config)

    owner = config['owner']
    paths = config['paths']
    imagestats = config['ImageStats']
    sbatch_kwargs = config['sbatch']
    preprocessing = config['preprocessing']
    methods = config['methods']
    evaluation = config['evaluation']

    if 'mail' in owner:
        mail = owner['mail']
    else:
        mail = None

    home = Path(paths['home'])
    data_path = Path(paths['data_path'])
    sample_name = paths['sample_name']
    gt_path = Path(paths['gt_path'])
    # define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = Path(paths['sections_path'])
    else:
        sections_path = processed / 'sections_px.json'

    with open(sections_path) as f:
        section_dictionary = json.load(f)
        sections = section_dictionary.keys()

    # define processed and results directory
    processed = Path(f'{home}/{sample_name}/processed/')
    processed.mkdir(parents=True, exist_ok=True)
    results = Path(f'{home}/{sample_name}/results/')
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample_name}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    job_dir = Path(f'{home}/{sample_name}/run/jobs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    sbatch_kwargs.update(
        log_path=str(log_path),
        job_dir=str(job_dir),
        mail=mail,
    )

    variables.update(dict(
        home=home,
        data_path=data_path,
        sample_name=sample_name,
        sections_path=sections_path,
        sections=sections,
        processed=processed,
        results=results,
        sbatch_kwargs=sbatch_kwargs,
        imagestats=imagestats,
        section_dictionary=section_dictionary
    ))

    if method in methods:
        results = Path(results / f'{method}/output/')
        results.mkdir(parents=True, exist_ok=True)
        variables.update(dict(
            method=methods[method],
            results=results,
            pixelsizeXY=imagestats['pixelsize_xy'],
            planes=preprocessing['planes']
        ))
    else:
        if method == 'eval':
            variables.update(dict(
                gt_path=gt_path,
                methods=methods,
                PD=evaluation['pd'],
                PCA=evaluation['pca'],
                JACCARD=evaluation['jaccard'],
                CS_BENCH=evaluation['cs_bench'],
            ))
        elif method == 'cross':
            variables.update(dict(
                methods=methods,
                CROSS=evaluation['cross']
            ))
        elif method in [
            'transcripts',
            'images',
            'boundaries'
        ]:
            variables.update(dict(
                preprocessing=preprocessing,
                pixelsizeXY=imagestats['pixelsize_xy'],
                pixelsizeZ=imagestats['pixelsize_z'],
            ))
        elif method == 'main':
            variables.update(dict(
                tasks=config['tasks'],
                PD=evaluation['pd']['use'],
                PCA=evaluation['pca']['use'],
                JACCARD=evaluation['jaccard']['use'],
                CS_BENCH=evaluation['cs_bench']['use'],
                CROSS=evaluation['cross']['use']
            ))

    return variables


# function form cellpose.utils
def outlines_list(masks, multiprocessing_threshold=1000, multiprocessing=None):
    '''Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): Array of masks.
        multiprocessing_threshold (int, optional):
            Threshold for enabling
            multiprocessing. Defaults to 1000.
        multiprocessing (bool, optional):
            Flag to enable multiprocessing. Defaults to None.

    Returns:
        list: List of outlines.

    Raises:
        None

    Notes:
        - This function is a wrapper for outlines_list_single and
          outlines_list_multi.
        - Multiprocessing is disabled for Windows.
    '''
    # default to use multiprocessing if not few_masks,
    # but allow user to override
    if multiprocessing is None:
        few_masks = np.max(masks) < multiprocessing_threshold
        multiprocessing = not few_masks
    # disable multiprocessing for Windows
    if os.name == "nt":
        if multiprocessing:
            logging.getLogger(__name__).warning(
                "Multiprocessing is disabled for Windows")
        multiprocessing = False
    if multiprocessing:
        print('  - Using Multiprocessing')
        return outlines_list_multi(masks)
    else:
        return outlines_list_single(masks)


# function form cellpose.utils
def outlines_list_single(masks):
    '''Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    '''
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(
                mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix


# function form cellpose.utils
def outlines_list_multi(masks, num_processes=None):
    '''Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.
    '''
    if num_processes is None:
        num_processes = cpu_count()
    unique_masks = np.unique(masks)[1:]
    with Pool(processes=num_processes) as pool:
        outpix = pool.map(
            get_outline_multi,
            [(masks, n) for n in unique_masks]
        )
    return outpix


# function form cellpose.utils
def get_outline_multi(args):
    '''Get the outline of a specific mask in a multi-mask image.

    Args:
        args (tuple): A tuple containing the masks and the mask number.

    Returns:
        numpy.ndarray: The outline of the specified mask as an array
                       of coordinates.

    '''
    masks, n = args
    mn = masks == n
    if mn.sum() > 0:
        contours = cv2.findContours(
            mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE
        )
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        return pix if len(pix) > 4 else np.zeros((0, 2))
    return np.zeros((0, 2))


# function form stackoverflow
# adapted to return shapely Polygons
def mask_to_polygons(npy_data, output_path):
    '''Get the polgyons from the prediction-masks using cellpose.utils functions
    Saves them as a GeoDataFrame (geojson)

    Parameters
    ----------
        npy_data : ArrayLike
            The numpy.ndarray of the masks.
        npy_base_output_path : Path
            Path to save the geojson.

    Returns
    ----------
        out : None
            Automatically saves the GDF under npy_base_output_path.
    '''
    print(' - Extracting ROI')
    try:
        masks = npy_data.item().get("masks")
    except (AttributeError, ValueError) as e:
        masks = npy_data
    masks = masks.squeeze()
    # change the index order:
    # first the cell then the layer it is on.
    # thus one would now how the same cell looks on different layers
    data = {'layer': [], 'name': [], 'geometry': []}
    if masks.ndim == 3:
        for z in range(masks.shape[0]):
            print(f' - Layer {z}')
            coords_list = outlines_list(masks[z, :, :])
            i = 1
            for coords in coords_list:
                data['layer'].append(z)
                data['name'].append(f'cell_{i}')
                data['geometry'].append(Polygon(coords))
                i += 1
    else:
        coords_list = outlines_list(masks)
        i = 1
        for coords in coords_list:
            data['layer'].append(np.nan)
            data['name'].append(f'cell_{i}')
            data['geometry'].append(Polygon(coords))
            i += 1
    gdf = gpd.GeoDataFrame(data=data)
    gdf.set_index(['layer', 'name'])
    print(' - Saving GeoDataFrame')
    gdf.to_file(output_path, driver='GeoJSON', index=True)
    return gdf
