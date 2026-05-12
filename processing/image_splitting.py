
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

from pathlib import Path
import configparser
import argparse
import os

from tqdm import tqdm
import psutil
import json

from tifffile import imread, imwrite, TiffWriter, TiffFile
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import zarr
import cv2


def get_memory_usage_percentage():
    process = psutil.Process()
    total_memory = psutil.virtual_memory().total  # Total system memory in bytes
    mem_info = process.memory_info()
    used_memory = mem_info.rss  # Resident Set Size in bytes
    memory_percentage = (used_memory / total_memory) * 100  # Calculate percentage
    return memory_percentage


def find_rois(contours, n_roi):
    """Sort the contours by area.
    Args:
        contours: The contours from cv2.findContours.
        n_roi: Expected # of regions of interest.
               Should be equivalent to the number of tissue-samples on the slide. 
    Returns:
        Contours of significant size.
    """

    values = []
    dtype = [('area', float), ('y', float), ('x', float)]

    for c in contours:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        values.append((w*h, y, x))

    values_arr = np.array(values, dtype=dtype)
    values_arr_sorted = np.sort(values_arr, kind='stable', order='area')
    smallest_allowed_roi = values_arr_sorted['area'][-n_roi]

    # sort contours by y,x:
    # values_nroi_argsorted = np.argsort(
    #     values_nroi,
    #     kind='stable',
    #     order=['y', 'x']
    # )

    mask = values_arr['area'] >= smallest_allowed_roi
    nroi_contours = [contours[index] for index, boolean in enumerate(mask) if boolean]

    return nroi_contours


def tif_path(section, ome=True, focus=False, chunk=None, layer=None):
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

    processed = config['PATHS'].get('processed')
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

    dir_path = Path('/'.join([processed, f_str]))
    dir_path.mkdir(parents=True, exist_ok=True)

    f_str = '/'.join([f_str, ext])
    file_path = Path('/'.join([processed, f_str]))

    return file_path


def get_window_shape(image, chunks, overlap):
    shape = image.shape
    if shape[-1] == 4:
        y, x, c = shape
        window_shape = (
            int(y*(np.sqrt(chunks)/chunks*(1+overlap))),
            int(x*(np.sqrt(chunks)/chunks*(1+overlap))),
            1
        )
    else:
        z, y, x = shape
        window_shape = (
            1,
            int(y*(np.sqrt(chunks)/chunks*(1+overlap))),
            int(x*(np.sqrt(chunks)/chunks*(1+overlap)))
        )
    return window_shape


def get_strides(image, window_shape):
    if window_shape[-1] == 1:
        y, x, c = image.shape
        y_w, x_w, c_w = window_shape
    else:
        z, y, x = image.shape
        z_w, y_w, x_w = window_shape

    stride_y = y-y_w
    stride_x = x-x_w

    return (stride_y, stride_x)

def view(image, chunks, shape, overlap):
    """
    Args:
        image: np.array of the image section
        chunks: how many chunks the image should end up in
        shape: the pre-chunks image-section shape
    Return:
        An array of shape (chunk, layers, y, x)
    """
    window_shape = get_window_shape(image, chunks, overlap)
    strides = get_strides(image, window_shape)
    if shape[-1] == 4:
        view_image = sliding_window_view(image, window_shape)
        view_shape = view_image.shape

        view_image = view_image[
            ::strides[0],
            ::strides[1],
            ...
        ]
        view_image = np.reshape(
            view_image,
            (chunks,)+window_shape[:-1]+(4,)
        )
    else:
        view_image = sliding_window_view(image, window_shape)
        view_shape = view_image.shape

        # this step needs to be changed so it can work with
        # any amount of chunks
        view_image = view_image[
            :,
            ::strides[0],
            ::strides[1],
            ...
        ]
        # print(type(view_image))
        # print(z, type(z))
        # print(chunks, type(chunks))
        # print(window_shape, type(window_shape))
        view_image = np.reshape(
            view_image,
            (z, chunks) + window_shape[1:]
        )

    return view_image


def write_tif(image, imagestats, section, layer=None, chunk=None):
    """Write an array into a tif file.

    Args:
        image: numpy.ndarray of the image.
        section: roi corresponding to image.
        config: parsed config
        layer: the layer of the morphology image. Passed on to tif_path.
        chunk: chunk corresponding to image. Passed on to tif_path.

    Retruns:
        None. Saves file under
        'processed/
        {section}/
        morphology/
        {focus or multi_layer or single_layer/layer0{layer}}/
        {quatered/q0{chunk}.extension if chunk, 
         else focus. or morphology.extension}'
    """
    pixelsizeXY = imagestats['pixelsizeXY']
    pixelsizeZ = imagestats['pixelsizeZ']

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
        subresolutions = 2
        bigtiff = True
        metadata = {
            'axes': axes,
            'PhysicalSizeX': pixelsizeXY,
        #    'PhysicalSizeXUnit': 'Ã‚Âµm',
            'PhysicalSizeY': pixelsizeXY,
        #    'PhysicalSizeYUnit': 'Ã‚Âµm',
            'PhysicalSizeZ': pixelsizeZ,
        #    'PhysicalSizeZUnit': 'Ã‚Âµm'
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
        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        if ome:
            for level in range(subresolutions):
                mag = 2 ** (level + 1)

                if focus:
                    image_ = image[::mag, ::mag, ...]
                else:
                    image_ = image[..., ::mag, ::mag]

                tif.write(
                    image_,
                    subfiletype=1,
                    resolution=(resolution[0] // mag,
                                resolution[1] // mag
                    ),
                    **options
                )

            # add a thumbnail image as a separate series
            # it is recognized by QuPath as an associated image
            thumbnail = (image[0, ::16, ::16] >> 2).astype('uint8')
            tif.write(thumbnail, metadata={'Name': 'thumbnail'})


def parse_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    preprocessing = dict(config.items('PREPROCESSING'))
    paths = dict(config.items('PATHS'))

    return preprocessing, paths



if __name__ == '__main__':

    # define paths
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    config = configparser.ConfigParser()
    config.read(config_path)

    preprocessing = dict(config.items('PREPROCESSING'))
    paths = dict(config.items('PATHS'))
    imagestats = dict(config.items('ImageStats'))

    # define paths
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']

    processed = Path('{0}/{1}/processed'.format(home, sample))
    processed.mkdir(parents=True, exist_ok=True)

    # load morpho and focus:
    morphology_store = imread(f'{data}/morphology.ome.tif', aszarr=True)
    morphology_zarr = zarr.open(morphology_store, mode='r')

    # load morphology_focus
    focus_org = []
    for file in Path(f'{data}/morphology_focus').glob('*.ome.tif'):
        focus_store = imread(file,
            aszarr=True,
            is_ome=False # to prevent multifile reading
        )
        focus_zarr = zarr.open(focus_store, mode='r')
        focus_org.append(focus_zarr['0'])

    subres_lvls = [lvl for lvl in morphology_zarr]
    subres_max = max(subres_lvls)
    subres_min = min(subres_lvls)

    morphology_org = morphology_zarr[subres_min]

    if 'section_dict' in paths:
        print('Has sections_dict...') 
        with open(paths['sections_dict'], 'r') as f:
            sections_dict = json.load(f)
        print('Coordinates loaded...')
    else:
        print('Searching for ROIs...')
        sections_dict = {}

        morphology_subres = morphology_zarr[subres_max]
        z, y, x = morphology_org.shape
        z_, y_, x_ = morphology_subres.shape

        rf_x = int(x/x_)
        rf_y = int(y/y_)

        subres_centre = np.uint8(morphology_subres[z//2])
        # subres_blur = cv2.GaussianBlur(
        #     subres_centre,
        #     (0, 0),
        #     1.5
        # )
        subres_dilated = cv2.dilate(
            subres_centre,
            np.ones((3, 3)),
            iterations=5
        )
        _, subres_binary = cv2.threshold(
            subres_dilated,
            127, 255, 0
        )
        imwrite('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/subres7_dil.tif', subres_binary)
        contours, _ = cv2.findContours(
            subres_binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # keep contours with significant size
        roi_list = find_rois(contours, preprocessing['n_roi'])

        buffer = preprocessing['buffer']

        for section, contour in enumerate(tqdm(
            roi_list,
            desc='Saving Coordinates',
            ncols=79,
            leave=True
        )):
            # add roi to scaled image to check for regions
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                subres_centre, (x, y), (x+w, y+h),
                (255, 255, 255), 2
            )

            # adjust for scaling
            x, w = x*rf_x, w*rf_x
            y, h = y*rf_y, h*rf_y
            # add buffer
            x_min, y_min = int(x*(1-buffer)), int(y*(1-buffer))
            x_max, y_max = int((x+w)*(1+buffer)), int((y+h)*(1+buffer))
            # add to dictionary
            sections_dict[str(section)] = [
                [y_min, x_min],
                [y_max, x_max]
            ]

        # save selected regions
        with open(processed / 'sections_px.json', 'w') as f:
            json.dump(sections_dict, f)

        imwrite(
            processed / 'marked_regions-of-interest.tif',
            subres_centre
        )

    with tqdm(
        total=len(sections_dict),
        desc='Saving ROIs',
        ncols=79,
        leave=True
    ) as section_bar:

        planes = preprocessing['planes']
        planes = [int(n) for n in planes if n.isdigit()]
        
        for section, bbox in sections_dict.items():
            y_min, x_min = bbox[0]
            y_max, x_max = bbox[1]
            resolution = (y_max-y_min, x_max-x_min)

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

            write_tif(morphology_section, imagestats, section)
            write_tif(focus_section, imagestats, section)

            z, y, x = morphology_section.shape
            for l, plane in enumerate(planes):
                write_tif(
                    morphology_section[l, ...], imagestats,
                    section, layer=plane
                )
                
                memory_percentage = get_memory_usage_percentage()
                section_bar.set_description(
                    f'Saving ROIs | %MEM: {memory_percentage:.2f}'
                )

            view_morphology = view(
                morphology_section, preprocessing['chunks'],
                morphology_section.shape, preprocessing['overlap']
            )
            view_focus = view(
                focus_section, preprocessing['chunks'],
                focus_section.shape, preprocessing['overlap']
            )

            with tqdm(
                total=chunks,
                desc='saving as chunks',
                ncols=79,
                leave=False
            ) as chunk_bar:
                for chunk in range(chunks):
                    q_m = view_morphology.copy()[:, chunk, ...]
                    q_f = view_focus.copy()[chunk, ...]
                    
                    write_tif(
                        q_m, imagestats, section, chunk=chunk
                    )
                    write_tif(
                        q_f, imagestats, section, chunk=chunk
                    )

                    for l, plane in enumerate(planes):
                        write_tif(q_m[l, ...], section,
                                    chunk=chunk, layer=plane)
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