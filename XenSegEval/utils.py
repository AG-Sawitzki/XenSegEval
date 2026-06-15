import os
import gzip
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import tifffile
import tomlkit
from shapely.geometry import Polygon
import geopandas as gpd
import cv2

# types
from geopandas.geodataframe import GeoDataFrame
from numpy.typing import ArrayLike
from typing import Any, Union


def polygon_to_mask(
    gdf: Union[str, os.PathLike, GeoDataFrame],
    shape: tuple,
    layer: int,
) -> ArrayLike:
    '''GeoJson Polygons to masks in a TIF.
    Args:
        gdf: path to geojson(.gz) or geodataframe.
        output_path: path to output location. might not be necessary.
    Retruns:
        Masks in numpy-array.
    '''
    if Path(gdf).suffix == '.gz':
        with gzip.open(gdf) as file:
            gdf = gpd.read_file(file)
    elif Path(gdf).suffix == 'geojson':
        gdf = gpd.read_file(gdf)
    elif type(gdf) is GeoDataFrame:
        gdf = gdf
    else:
        print('gdf not path or GeoDataFrame.')

    r, g, b = (0,)*3
    img = np.zeros(shape, np.uint8)
    for mpg in gdf[gdf['layer']==layer]['geometry']:
        for lr in mpg.geoms:
            pl = np.array(list(lr.exterior.coords))
            cv2.fillPoly(img, np.int32([pl]), (r,g,b))
            if r < 255:
                r+=1
            else:
                if g < 255:
                    g+=1
                    r=0
                else:
                    if b < 255:
                        b+=1
                        g=0
                        r=0
                    else:
                        print('no colours left')
    return img



# function form cellpose.utils
def outlines_list(masks, multiprocessing_threshold=1000, multiprocessing=None):
    '''Get outlines of masks as a list to loop over for plotting.
    Args:
        masks (ndarray): Array of masks.
        multiprocessing_threshold (int, optional): Threshold for enabling multiprocessing. Defaults to 1000.
        multiprocessing (bool, optional): Flag to enable multiprocessing. Defaults to None.
    Returns:
        list: List of outlines.
    Raises:
        None
    Notes:
        - This function is a wrapper for outlines_list_single and outlines_list_multi.
        - Multiprocessing is disabled for Windows.
    '''
    # default to use multiprocessing if not few_masks, but allow user to override
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
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
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
        outpix = pool.map(get_outline_multi, [(masks, n) for n in unique_masks])
    return outpix


# function form cellpose.utils
def get_outline_multi(args):
    '''Get the outline of a specific mask in a multi-mask image.
    Args:
        args (tuple): A tuple containing the masks and the mask number.
    Returns:
        numpy.ndarray: The outline of the specified mask as an array of coordinates.

    '''
    masks, n = args
    mn = masks == n
    if mn.sum() > 0:
        contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_NONE)
        contours = contours[-2]
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()
        return pix if len(pix) > 4 else np.zeros((0, 2))
    return np.zeros((0, 2))


# function form stackoverflow
# adapted to return shapely Polygons
def process_roi(npy_data, output_path):
    '''Prediction-mask to polygons in GeoDataFrame (geojson) using Cellpose.utils.  
    Args:
        npy_data: The numpy.ndarray of the masks.
        npy_base_output_path: Path to save the geojson.
    Returns:
        Nothing. Automatically saves the GDF.
    '''
    print(' - Extracting ROI')
    try:
        masks = npy_data.item().get("masks")
    except:
        masks = npy_data
    masks = masks.squeeze()
    # change the index order:
    # first the cell then the layer it is on.
    # thus one would now how the same cell looks on different layers
    data = {'layer': [], 'name': [], 'geometry': []}
    if masks.ndim == 3:
        for z in range(masks.shape[0]):
            print(f' - Layer {z}')
            coords_list = outlines_list(masks[z,:,:])
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
    gdf = gpd.GeoDataFrame(data = data)
    gdf.set_index(['layer', 'name'])
    print(' - Saving GeoDataFrame')
    gdf.to_file(output_path, driver='GeoJSON', index = True)


def submit_sbatch(
    tempfile_dir: Union[str, os.PathLike[Any]],
    time: int,
    mem: int,
    cpu: int,
    log_path: Union[str, os.PathLike[Any]],
    cmd: str,
    gpu: Union[str, None] = None,
    mail: Union[str, None] = None,
) -> str:
    '''Writes a job-file for sbatch akd retruns the command to submit it.
    Args:
        tempfile_dir: path to a directory 
                      where the sbatch files will be saved.
        time: days to reserve the node for.
        mem: how much RAM to request.
        cpu: how many cpu-cores to request.
        log_path: path to directory
                  where the logs will be saved.
        cmd: the command to run on the node.
        gpu(optional): wether to run on a gpu node or not.
        mail(optional): the mail-address to send sbatch updates to.
    Returns:
        string with which the job can be submitted
    '''
    if cmd.partition(' ')[0] == 'bash':
        file = cmd[cmd.find('Xen'):cmd.rfind('.sh')]
        name = Path(file).stem
    else:
        file = cmd[cmd.find('Xen'):cmd.find(' -c')]
        name = file.rpartition('.')
        name = name[-1]
        name.replace('-','')
        if name == 'eval':
            name += '_'+cmd[cmd.rfind('-m')+3:]
    with open(f'{tempfile_dir}/{name}.sh', 'w+') as fh:
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
            fh.writelines(f'#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT_80,TIME_LIMIT_50\n')
        fh.writelines('#\n')
        fh.writelines('. ~/.bashrc\n')
        fh.writelines('export PIXI_CACHE_DIR=~/scratch/.cache/pixi')
        fh.writelines('\n#\n')
        fh.writelines(f'pixi run {cmd}\n')

    return f'sbatch {fh.name}'


def get_config_args(
    config: Union[str, os.PathLike[Any], dict],
    method: Union[str, None] = None
) -> dict:
    '''Return config
    Args:
        config: string or path to config.toml or dict of parsed config.
        method: string of pipeline step.
    Returns:
        dictionary of parsed config.
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
    ## define sections_dictionary path
    if 'sections_path' in paths:
        sections_path = Path(paths['sections_path'])
    else:
        sections_path = processed / 'sections_px.json'
    
    ## define processed and results directory
    processed = Path(f'{home}/{sample_name}/processed/')
    processed.mkdir(parents=True, exist_ok=True)
    results = Path(f'{home}/{sample_name}/results/')
    results.mkdir(parents=True, exist_ok=True)
    log_path = Path(f'{home}/{sample_name}/run/logs/')
    log_path.mkdir(parents=True, exist_ok=True)
    tempfile_dir = Path(f'{home}/{sample_name}/run/jobs/')
    tempfile_dir.mkdir(parents=True, exist_ok=True)

    sbatch_kwargs.update(
        log_path=str(log_path),
        tempfile_dir=str(tempfile_dir),
        mail=mail,
    )

    variables.update(dict(
        home=home,
        data_path=data_path,
        sample_name=sample_name,
        sections_path=sections_path,
        processed=processed,
        results=results,
        sbatch_kwargs=sbatch_kwargs,
        imagestats=imagestats
    ))

    if method in methods:
        results = Path(results / f'{method}/output/')
        results.mkdir(parents=True, exist_ok=True)
        variables.update(dict(
            method=methods[method],
            results=results,
            pixelsizeXY = imagestats['pixelsize_xy']
        ))
    else:
        if method == 'eval':
            variables.update(dict(
                gt_path=gt_path,
                PD=evaluation['PD'],
                PCA=evaluation['PCA'],
                JACCARD=evaluation['JACCARD'],
                CS_BENCH=evaluation['CS-BENCH'],
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
                tasks=config['Tasks']
            ))

    return variables