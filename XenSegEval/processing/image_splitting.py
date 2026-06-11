
'''
Segment the sample by regions of interest.
Saves the coordinates of top left and bottom right corner in a dictionary.
Unit is px.

Theoretically NMS should be added... !!!

ToDo:
    +- add path variability
        +- preferably with checks for existence
    +- config-file compatibility?
    + adaptable roi size?
        (- by checking how many are of sufficient size?!)
'''

from itertools import product
from pathlib import Path
import configparser
import argparse
import os

from tqdm import tqdm
import tomlkit
import psutil
import json

from tifffile import imread, imwrite, TiffWriter, TiffFile
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
import numpy as np
import zarr
import cv2

print(os.getcwd())


def get_memory_usage_percentage() -> float:
    """Get the memory usage as percantage.
    Returns:
        Float of currently used memory.
    """
    process = psutil.Process()
    # Total system memory in bytes
    total_memory = psutil.virtual_memory().total
    # Resident Set Size in bytes
    mem_info = process.memory_info()
    used_memory = mem_info.rss
    memory_percentage = (used_memory / total_memory) * 100  # Calculate percentage
    return memory_percentage


def tif_path(
    section: str,
    ome: bool=True,
    focus: bool=False,
    chunk: int=None,
    layer: int=None
) -> Path:
    """Create the path to the tif file.
    Args:
        Section: Which sample on the slide is examined.
        p_processed: The path so the 'processed' directory.
        ome: Boolean. If file will be ome or not.
        focus: Boolean. If file contains channels or layers.
        chunk: the chunk of the section.
        layer: ome-layer of the source image.
    Returns:
        pathlib.Path
    """

    f_str = '/'.join([str(section), 'morphology'])

    if ome:
        if focus:
            f_str = '/'.join([f_str, 'focus'])
        else:
            f_str = '/'.join([f_str, 'multi_layer'])
        ext = 'ome.tif'
    else:
        f_str = '/'.join([f_str, f'single_layer/layer0{layer}'])
        ext = 'tif'
    if chunk is not None:
        f_str = '/'.join([f_str, 'quatered'])
        ext = '.'.join([f'q0{chunk}', ext])
    else:
        if focus:
            ext = '.'.join(['focus', ext])
        else:
            ext = '.'.join(['morphology', ext])

    dir_path = Path(processed / f_str) # Path('/'.join([processed, f_str]))
    dir_path.mkdir(parents=True, exist_ok=True)

    f_str = '/'.join([f_str, ext])
    file_path = Path(processed / f_str)
    return file_path


def chunk_size(
    var: int,
    chunks: int
) -> int:
    """Calculate the size of a chunk of a region.
    Args:
        var: Total region width or height.
        chunks: Total numbr of chunks.
    Returns:
        Width or Height of chunk.
    """
    return int(var*np.sqrt(chunks)/chunks)


def write_tif(
    image: ArrayLike,
    imagestats: dict,
    section: str | int,
    layer: int=None,
    chunk: int=None
) -> None:
    """Write an array into a tif file.
    Args:
        image: numpy.ndarray of the image.
        section: ROI name.
        imagestats: Dictionary containing stats of the image.
        layer: The layer of the morphology image being written. 
               Passed on to tif_path.
        chunk: Chunk corresponding to image being written. 
               Passed on to tif_path.
    Retruns:
        None. Saves file under
        'processed/{section}/morphology/
        {focus or multi_layer or single_layer/layer0{layer}}/
        {quatered/q0{chunk}.extension if chunk
         else focus. or morphology.extension}'
    """
    try:
        pixelsizeXY = imagestats['pixelsize_xy']
        pixelsizeZ = imagestats['pixelsize_z']
    except:
        print('Imagestats are missing "pixelsize_xy" | "pixelsize_z".')
    
    options = dict(
        compression=None,
        resolutionunit='MICROMETER'
    )

    if image.ndim == 3:
        ome = True
        focus = False
        axes = 'ZYX'
        resolution = image.shape[1:]
        if image.shape[-1] == 4:
            focus = True
            axes = 'YXC'
            resolution = image.shape[:2]
        subresolutions = None
        bigtiff = True
        metadata = {
            'axes': axes,
            'PhysicalSizeX': pixelsizeXY,
            'PhysicalSizeXUnit': 'Âµm',
            'PhysicalSizeY': pixelsizeXY,
            'PhysicalSizeYUnit': 'Âµm',
            'PhysicalSizeZ': pixelsizeZ,
            'PhysicalSizeZUnit': 'Âµm'
        }
    else:
        ome = False
        focus = False
        axes = 'YX'
        resolution = image.shape
        subresolutions = None
        bigtiff = False
        metadata = None

    file = tif_path(section, ome, focus, chunk, layer)

    with TiffWriter(file, bigtiff=bigtiff) as tif:

        tif.write(
            image,
            subfiletype=None,
            subifds=subresolutions,
            resolution=resolution,
            metadata=metadata,
            **options
        )


if __name__ == '__main__':
    # define paths
    parser = argparse.ArgumentParser(
        prog='img',
        description='Split morphology and focus image into defined sections.')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    preprocessing = config['preprocessing']
    paths = config['paths']
    imagestats = config['ImageStats']

    # define paths
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']
    ## define processed directory    
    processed = Path(f'{home}/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)
    ## define sections_dictionary path
    sections_path = paths['sections_path']

    with open(sections_path, 'r') as f:
        sections_dict = json.load(f)

    # load morpho and focus:
    morphology_store = imread(f'{data}/morphology.ome.tif', aszarr=True)
    morphology_zarr = zarr.open(morphology_store, mode='r')

    subres_lvls = [lvl for lvl in morphology_zarr]
    subres_max = max(subres_lvls)
    subres_min = min(subres_lvls)

    morphology_org = morphology_zarr[subres_min]

    # load morphology_focus
    focus_org = []
    for file in Path(f'{data}/morphology_focus').glob('*.ome.tif'):
        focus_store = imread(file,
            aszarr=True,
            is_ome=False # to prevent multifile reading
        )
        focus_zarr = zarr.open(focus_store, mode='r')
        focus_org.append(focus_zarr['0'])

    with tqdm(
        total=len(sections_dict),
        desc='Saving ROIs',
        ncols=79,
        leave=True
    ) as section_bar:

        planes = preprocessing['planes']
        # planes = [int(n) for n in planes if n.isdigit()]
        chunks = preprocessing['chunks']
        overlap = preprocessing['overlap']

        for section, bbox in sections_dict.items():
            y_min, x_min = bbox[0]
            y_max, x_max = bbox[1]
            resolution = (y_max-y_min, x_max-x_min)

            # assigning the arrays take ~4min
            morphology_section = morphology_org[
                planes,
                y_min:y_max,
                x_min:x_max
            ]
            focus_section = np.dstack((
                focus_org[0][y_min:y_max, x_min:x_max],
                focus_org[1][y_min:y_max, x_min:x_max],
                focus_org[2][y_min:y_max, x_min:x_max],
                focus_org[3][y_min:y_max, x_min:x_max]
            ))
            # print(type(focus_section))
            # print(type(morphology_section))

            write_tif(morphology_section, imagestats, section)
            write_tif(focus_section, imagestats, section)

            for l, plane in enumerate(planes):
                write_tif(
                    morphology_section[l, ...],
                    imagestats, section, layer=plane
                )
                
                memory_percentage = get_memory_usage_percentage()
                section_bar.set_description(
                    f'Saving ROIs | %MEM: {memory_percentage:.2f}'
                )

            with tqdm(
                total=chunks,
                desc='saving as chunks',
                ncols=79,
                leave=False
            ) as chunk_bar:

                z, y, x = morphology_section.shape

                y_size = chunk_size(y, chunks)
                x_size = chunk_size(x, chunks)

                grid = product(
                    range(0, y-y%y_size, y_size),
                    range(0, x-x%x_size, x_size)
                )

                for chunk, (y_c, x_c) in enumerate(grid):
                    y_low = int(y_c*0.95)
                    y_high = int((y_c+y_size)*1.05)

                    x_low = int(x_c*0.95)
                    x_high = int((x_c+x_size)*1.05)

                    morphology_chunk = morphology_section[
                        :, y_low:y_high, x_low:x_high
                    ]
                    focus_chunk = focus_section[
                        y_low:y_high, x_low:x_high, :
                    ]

                    write_tif(
                        morphology_chunk, imagestats, section, chunk=chunk
                    )
                    write_tif(
                        focus_chunk, imagestats, section, chunk=chunk
                    )

                    for l, plane in enumerate(planes):
                        write_tif(
                            morphology_chunk[l, ...], imagestats,
                            section, chunk=chunk, layer=plane
                        )

                    memory_percentage = get_memory_usage_percentage()
                    chunk_bar.set_description(
                        f'saving as chunks | %MEM: {memory_percentage:.2f}'
                    )
                    chunk_bar.update(1)
            
            memory_percentage = get_memory_usage_percentage()
            section_bar.set_description(
                f'Saving ROIs | %MEM: {memory_percentage:.2f}'
            )
            section_bar.update(1)