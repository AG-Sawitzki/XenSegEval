from XenSegEval.utils import get_config_args

from itertools import product
from pathlib import Path
import configparser
import argparse
import sys
import os

from tqdm import tqdm
import tomlkit
import psutil
import json

from tifffile import imread, imwrite, TiffWriter, TiffFile
import numpy as np
import zarr
import cv2
# from aicsimageio.writers import ome_tiff_writer

# types
from numpy.typing import ArrayLike
from typing import Any, Union


def get_memory_usage_percentage() -> float:
    """Get the memory usage as percentage.

    Returns
    ----------
        out : float
            Float of currently used memory in percentage.
    """
    process = psutil.Process()
    # Total system memory in bytes
    total_memory = psutil.virtual_memory().total
    # Resident Set Size in bytes
    mem_info = process.memory_info()
    used_memory = mem_info.rss
    # Calculate percentage
    memory_percentage = (used_memory / total_memory) * 100
    return memory_percentage


def chunk_size(
    var: int,
    chunks: int
) -> int:
    """Calculate the size of a chunk of a region.

    Parameters
    ----------
        var : int 
            Total region width or height.
        chunks : int
            Total numbr of chunks.

    Returns
    ----------
        out : int
            Width or Height of chunk.
    """
    return int(var*np.sqrt(chunks)/chunks)


def tif_path(
    section: str,
    ome: bool = True,
    focus: bool = False,
    chunk: int = None,
    layer: int = None
) -> Path:
    """Create the path to the tif file.

    Parameters
    ----------
        section : str
            Which sample on the slide is examined.
        ome : bool, optional
            If file will be ome or not.
            Default is `True`.
        focus : bool
            If file contains channels or layers.
            Default is `False`.
        chunk : int, optional
            The chunk of the section.
            Default is `None`.
        layer : int, optional
            OME-layer of the source image.
            Default is `None`.

    Returns:
    ----------
        out : PosixPath
            Path to the image.
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

    dir_path = Path(processed / f_str)
    dir_path.mkdir(parents=True, exist_ok=True)

    f_str = '/'.join([f_str, ext])
    file_path = Path(processed / f_str)

    return file_path


def write_tif(
    image: ArrayLike,
    imagestats: dict,
    section: Union[str, int],
    layer: int = None,
    chunk: int = None
) -> None:
    """Write an array into a tif file.

    Parameters:
    ----------
        image : ArrayLike
            numpy.ndarray of the image.
        imagestats : dict
            Dictionary containing stats of the image.
        section : str or int
            ROI name. The section the image represents.
        layer : int, optional
            The layer of the morphology image being written.
            Passed on to tif_path.
            Default is `None`.
        chunk : int
            Chunk corresponding to image being written.
            Passed on to tif_path.
            Default is `None`.

    Retruns
    ----------
        out : None
            Saves file under
            'processed/{section}/morphology/
            {focus or multi_layer or single_layer/layer0{layer}}/
            {quatered/q0{chunk}.extension if chunk
                else focus. or morphology.extension}'
    """
    if 'pixelsizeXY' not in globals():
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


def write_aics(
    image: ArrayLike,
    section: Union[str, int],
) -> None:
    '''Save focus-image for PCA analysis.

    Parameters
    ----------
        image : ArrayLike
            CYX array of the image.
        section : str or int
            ROI name. The section the image represents.

    Retruns
    ----------
        out : None
            Saves the image using AICSImageIO.
    '''
    image = np.moveaxis(image, -1, 0)

    writer = ome_tiff_writer.OmeTiffWriter()

    channel_names = [
        'DAPI',
        'ATP1A1_E-Cadherin_CD45',
        '18S_rRNA',
        'alphaSMA_Vimentin'
    ]

    stats = {
        'dim_order': 'CYX',
        'channel_names': channel_names,
        'image_name': 'focus',
        'pixel_physical_size': 0.2125,
        'channel_colours': ['red', 'green', 'blue', 'yellow']
    }

    file = tif_path(section, ome=True, focus=True)
    file = str(file).replace(file.stem, 'aics')
    writer.save(
        image,
        uri=file,
        **stats
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='IMGs')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'images')
    globals().update(variables)

    # load morpho and focus:
    morphology_store = imread(f'{data_path}/morphology.ome.tif', aszarr=True)
    morphology_zarr = zarr.open(morphology_store, mode='r')

    subres_lvls = [lvl for lvl in morphology_zarr]
    subres_max = max(subres_lvls)
    subres_min = min(subres_lvls)

    morphology_org = morphology_zarr[subres_min]

    # load morphology_focus
    focus_org = []
    focus_files = list(
        Path(f'{data_path}/morphology_focus').glob('*.ome.tif')
    )
    focus_files.sort()
    for file in focus_files:
        focus_store = imread(
            file,
            aszarr=True,
            is_ome=False  # to prevent multifile reading
        )
        focus_zarr = zarr.open(focus_store, mode='r')
        focus_org.append(focus_zarr['0'])

    with tqdm(
        total=len(section_dictionary),
        desc='Saving ROIs',
        ncols=79,
        leave=True
    ) as section_bar:

        planes = preprocessing['planes']
        # planes = [int(n) for n in planes if n.isdigit()]
        chunks = preprocessing['chunks']
        overlap = preprocessing['overlap']

        for section, bbox in section_dictionary.items():
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
            # if AICS is True:
            #     write_aics(focus_section, section)

            for L, plane in enumerate(planes):
                write_tif(
                    morphology_section[L, ...],
                    imagestats, section, layer=plane
                )

                memory_percentage = get_memory_usage_percentage()
                section_bar.set_description(
                    f'Saving ROIs | %MEM: {memory_percentage:.2f}'
                )

            if chunks > 0:
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
                        range(0, y-y % y_size, y_size),
                        range(0, x-x % x_size, x_size)
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

                        for L, plane in enumerate(planes):
                            write_tif(
                                morphology_chunk[L, ...], imagestats,
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
