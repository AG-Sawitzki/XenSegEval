
'''
Segment the sample by regions of interest.
Saves the coordinates of top left and bottom right corner in a dictionary.
Unit is px.

Theoretically NMS should be added...

ToDo:
    +- add path variability
        +- preferably with checks for existence
    - config-file compatibility?
    + adaptable roi size?
        (- by checking how many are of sufficient size?!)
'''

from pathlib import Path
import configparser
import argparse

import json

from tifffile import imread, imwrite, TiffWriter, TiffFile
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import cv2


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
    values_nroi = values_arr_sorted[-n_roi:]

    # sort contours by y,x:
    values_nroi_argsorted = np.argsort(values_nroi,
                                       kind='stable',
                                       order=['y', 'x'])
    contours_sorted = [contours[i] for i in values_nroi_argsorted]

    return contours_sorted


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
    f_str = '/'.join(str(section), '/morphology/')
    
    if ome:
        if focus:
            f_str = '/'.join(f_str, 'focus')
        else:
            f_str = '/'.join(f_str, 'multi_layer')
        ext = 'ome.tif'
    else:
        f_str = '/'.join(f_str, 'single_layer/layer0{0}'.format(layer))
        ext = 'tif'
    if chunk is not None:
        f_str = '/'.join(f_str, 'quatered/q0{0}'.format(chunk))

    path = Path(processed / f_str)
    path.mkdir(parents=True, exist_ok=True)

    f_str = '.'.join(f_str, ext)
    file = Path(processed / f_str)

    return file


def view(image, chunks, shape):
    """
    Args:
        image: np.array of the image section
        chunks: how many chunks the image should end up in
        shape: the pre-chunks image-section shape
    Return:
        An array of shape (chunk, layers, y, x)
    """
    if shape[-1] >= 4:
        y, x, c = shape
        window_shape = (int(y*(2/chunks+overlap)),
                        int(x*(2/chunks+overlap)),
                        1
        )

        view_image = sliding_window_view(image, window_shape)
        view_shape = view_image.shape

        view_image = view_image[::view_shape[1]-1,
                                ::view_shape[2]-1,
                                ...
        ]
        view_image = np.reshape(view_image,
                                (chunks,)+window_shape[:-1]+(4,)
        )
    else:
        z, y, x = shape
        window_shape = (1,
                        int(y*(2/chunks+overlap)),
                        int(x*(2/chunks+overlap))
        )
        view_image = sliding_window_view(image, window_shape)
        view_shape = view_image.shape

        view_image = sliding_window_view(image, window_shape)
        # this step needs to be changed so it can work with
        # any amount of chunks
        view_image = view_image[:,
                                ::view_shape[1]-1,
                                ::view_shape[2]-1,
                                ...
        ]
        view_image = np.reshape(view_image,
                                (z, chunks) + window_shape[1:]
        )

    return view_image


def write_tif(image, section, layer=None, chunk=None):
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
        {focus or multi_layer or single_layer/
                                 layer0{layer}}/
        {quatered/
        q0{chunk}.extension if chunk, else focus. or morphology.extension}'
    """
    pixelsizes = config['PREPROCESSING'].get('pixelsizes')
    options = dict(
            compression=None,
            resolutionunit='MICROMETER'
        )

    if image.ndim == 3:
        ome = True
        axes = 'ZYX'
        resolution = image.shape[1:]
        if image.shape[-1] >= 4:
            focus = True
            axes = 'YXC'
            resolution = image.shape[:3]
        subresolutions = 2
        bigtiff = True
        metadata = {
                'axes': axes,
                'PhysicalSizeX': pixelsizes[0],
                'PhysicalSizeXUnit': 'Âµm',
                'PhysicalSizeY': pixelsizes[0],
                'PhysicalSizeYUnit': 'Âµm',
                'PhysicalSizeZ': pixelsizes[1],
                'PhysicalSizeZUnit': 'Âµm'
        }
    else:
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
            if focus:
                image_ = image[::mag, ::mag, ...]
            else:
                image_ = image[..., ::mag, ::mag]
                # add a thumbnail image as a separate series
                # it is recognized by QuPath as an associated image
                thumbnail = (image[0, ::16, ::16] >> 2).astype('uint8')
                tif.write(thumbnail, metadata={'Name': 'thumbnail'})

            for level in range(subresolutions):
                mag = 2 ** (level + 1)
                tif.write(
                    image_,
                    subfiletype=1,
                    resolution=(resolution[0] / mag,
                                resolution[1] / mag
                                ),
                    **options
                    )


if __name__ == '__main__':

    # define paths
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    args = parser.parse_args()

    config_path = args.Config

    config = configparser.ConfigParser()
    config.read(config_path)

    # define variables
    chunks = config['PREPROCESSING'].getfloat('chunks')
    min_size = config['PREPROCESSING'].getfloat('min_size')
    n_roi = config['PREPROCESSING'].getfloat('n_roi')
    overlap = config['PREPROCESSING'].getfloat('overlap')
    pixelsizes = config['PREPROCESSING'].get('pixelsizes')
    layers = config['PREPROCESSING'].get('layers')
    buffer = config['PREPROCESSING'].getfloat('buffer')

    # define paths
    data = Path(config['PATHS']['data_path'])
    sample = config['PATHS']['sample_name']

    # processed = Path(f'/data/cephfs-2/unmirrored/groups/sawitzki/Juno/{sample}/processed')
    # processed.mkdir(parents=True, exist_ok=True)
    processed = config['PATHS'].get('processed')
    processed.mkdir(parents=True, exist_ok=True)

    # read morpho image
    if config.has_option('PATHS', 'sections_dict'):
        with open(config['PATHS']['sections_dict'], 'r') as f:
            sections_dict = json.load(f)
    else:
        sections_dict = {}
        with tiffFile(img_path) as tif:
            layers = len(tif.pages)
            centre_layer = int(layers//2)
            y, x = tif.pages[centre_layer].shape
            rf = int(y/1000)
            centre_page = tif.pages[centre_layer].asarray()
            # convert to CV_8UC1 compatible array,
            # downscale
            centre_page_scaled = np.uint8(centre_page[::rf, ::rf])
            # blur
            centre_page_scaled_blur = cv2.GaussianBlur(centre_page_scaled,
                                                       (0, 0), 1.5
            )
            # binary image
            ret, thresh = cv2.threshold(centre_page_scaled_blur,
                                        127, 255, 0
            )
            # dilate
            thresh_dilate = cv2.dilate(thresh,
                                       np.ones((5, 5)),
                                       iterations=3
            )
            # find contours
            contours, _ = cv2.findContours(thresh_dilate,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE
            )
            # keep contours with significant size
            roi_list, _ = find_rois(contours, n_roi)

            for section, c in enumerate(roi_list):
                # add roi to scaled image to check for regions
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(centre_page_scaled, (x, y), (x+w, y+h),
                              (255, 255, 255), 2
                )
                # adjust for scaling
                x, y, w, h = x*rf, y*rf, w*rf, h*rf
                sections_dict[str(section)] = [[y, x],
                                               [y+h, x+w]
                ]

        # save selected regions
        with open(processed / 'sections_px.json', 'w') as f:
            json.dump(sections_dict, f)

        imwrite(processed / 'marked_regions-of-interest.tif',
                centre_page_scaled
        )

    # crop images to sections of interest
    # additionally saves overlapping sub-sections
    with tiffFile(data / 'morphology.ome.tif') as mor, tiffFile(data / 'morphology_focus/morphology_focus_0000.ome.tif') as foc:

        layers = len(mor.pages)
        centre_layer = int(layers//2)
        morphology = np.vstack([mor.pages[i].asarray() for i in layers])
        focus = np.vstack([page.asarray() for page in foc.pagess])

        for section, bbox in sections_dict.items():

            y_min, x_min = bbox[0]
            y_max, x_max = bbox[1]
            resolution = (y_max-y_min, x_max-x_min)

            morphology_section = morphology[:,
                                            y_min:y_max,
                                            x_min:x_max
            ]
            write_tif(morphology_section, section)

            z, y, x = morphology_section.shape
            for layer in range(z):
                write_tif(morphology_section[layer, ...], section, layer=layer)

            focus_section = focus[y_min:y_max,
                                  x_min:x_max,
                                  :
            ]
            write_tif(focus_section, section)

            view_morphology = view(morphology_section, chunks,
                                   shape=morphology_section.shape)
            view_focus = view(focus_section, chunks, shape=focus_section.shape)
            for chunk in range(chunks):
                q_m = view_morphology.copy()[:, chunk, ...]
                q_f = view_focus.copy()[chunk, ...]
                if q.ndim != 3:
                    q = np.squeeze(q)
                    if q.ndim != 3:
                        print('something is weird')
                        break
                write_tif(q_m, section, chunk=chunk)
                write_tif(q_f, section, chunk=chunk)

                for layer in range(z):
                    write_tif(q_m[layer, ...], section,
                              chunk=chunk, layer=layer)
