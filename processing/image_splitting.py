
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
import numpy as np
import cv2


def find_rois(contours, n_roi):
    """
    Sort the contours by area
    Args:
        contours: The contours from cv2.findContours
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


def writeTif(image, section, chunk, layer, resolution, ome=True, focus=False):
    """
    Write the array into a tif file.

    Args:
        image: numpy.ndarray of the image.
        section: roi corresponding to image.
        chunk: chunk corresponding to image.
        layer: the layer of the morphology image.
        resolution: tuple of y,x pixel amount.
        ome: Boolean.
             True: the image is saved as '.ome.tif'
             False: the image is saved as '.tif'
        focus: Boolean. bound to 'ome'
               True: axes as YX + channel
               False: axes as ZYX

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

    if ome:
        axes = 'ZYX'
        if focus:
            axes = 'YXC'
        subresolutions = 2
        image_type = 'ome.tif'
        bigtiff = True
        metadata = {
                'axes': axes,
                'PhysicalSizeX': pixelsize,
                'PhysicalSizeXUnit': 'Âµm',
                'PhysicalSizeY': pixelsize,
                'PhysicalSizeYUnit': 'Âµm',
                'PhysicalSizeZ': 3.0000,
                'PhysicalSizeZUnit': 'Âµm'
            }
        layer = None
        path = Path(processed / 
                    '{0}/morphology/{1}/{2}'.format(section,
                                                    'focus' if focus else 'multi_layer',
                                                    'quatered/' if chunk is not None else ''))
    else:
        axes = 'YX'
        subresolutions = None
        image_type = 'tif'
        bigtiff = False
        metadata = None
        path = Path(processed / 
                    '{0}/morphology/single_layer/layer0{1}/{2}'.format(section,
                                                                       layer,
                                                                       'quatered/' if chunk is not None else ''
                                                                        )
                    )

    path.mkdir(parents=True, exist_ok=True)
    file = path / '{0}.{1}'.format('q0{chunk}'.format(chunk=chunk) if chunk is not None else 'morphology',
                                   image_type)
    with TiffWriter(file, bigtiff=bigtiff) as tif:

        metadata = metadata

        options = dict(
            compression=None,
            resolutionunit='MICROMETER'
        )

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

    data = Path(config['PATHS']['data_path'])
    sample = config['PATHS']['sample_name']

    processed = Path(f'/data/cephfs-2/unmirrored/groups/sawitzki/Juno/{sample}/processed')
    processed.mkdir(parents=True, exist_ok=True)

    # define variables
    chunks = config['PREPROCESSING'].getfloat('chunks')
    min_size = config['PREPROCESSING'].getfloat('min_size')
    n_roi = config['PREPROCESSING'].getfloat('n_roi')
    overlap = config['PREPROCESSING'].getfloat('overlap')
    pixelsize = config['PREPROCESSING'].getfloat('pixelsize')
    layers = config['PREPROCESSING'].get('layers')
    buffer = config['PREPROCESSING'].getfloat('buffer')

    # read morpho image
    if not config.has_option('PATHS', 'sections_dict'):
        sections_dict = {}
        with tiffFile(data / 'morpholog.ome.tif') as tif:
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
                                                       (0, 0),
                                                       1.5
                                                       )

            # binary image
            ret, thresh = cv2.threshold(centre_page_scaled_blur,
                                        127,
                                        255,
                                        0
                                        )

            # dilate
            thresh_dilate = cv2.dilate(thresh,
                                       np.ones((5, 5)),
                                       iterations=3
                                       )

            # rois are now nearly continuous shapes of similar brightness

            # find contours
            contours, _ = cv2.findContours(thresh_dilate,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE
                                           )

            # keep contours with significant size
            roi_list, _ = find_rois(contours, n_roi)

            section = 0
            for c in roi_list:

                # add roi to scaled image to check for regions
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(centre_page_scaled, 
                              (x, y),
                              (x+w, y+h),
                              (255, 255, 255),
                              2
                              )

                # adjust for scaling
                x, y, w, h = x*rf, y*rf, w*rf, h*rf
                sections_dict[str(section)] = [[y, x],
                                               [y+h, x+w]
                                               ]

                section += 1

        # save selected regions
        with open(processed / 'sections_px.json', 'w') as f:
            json.dump(sections_dict, f)

        imwrite(processed / 'marked_regions-of-interest.tif', centre_page_scaled)

    # crop images to sections of interest
    # additionally saves overlapping sub-sections
    with tiffFile(data / 'morphology.ome.tif') as tif:
        layers = len(tif.pages)
        centre_layer = int(layers//2)
        morphology = np.vstack([tif.pages[i].asarray() for i in layers])

        for section, bbox in sections_dict.items():

            y_min, x_min = bbox[0]
            y_max, x_max = bbox[1]
            resolution = (y_max-y_min, x_max-x_min)

            morphology_section = morphology[:,
                                            y_min:y_max,
                                            x_min:x_max
                                            ] # z,y,x
            focus_section = focus[y_min:y_max,
                                  x_min:x_max,
                                  :
                                  ]

            z, y, x = morphology_section.shape

            write_tif(morphology_section, section, resolution=resolution)
            write_tif(focus_section, section, resolution=resolution, focus=True)
            for layer in range(z):
                write_tif(morphology_section[layer, ...],
                          section,
                          layer=layer,
                          resolution=resolution,
                          ome=False
                          )
            #
            # sub-sections:
            window_shape_morphology = (1,
                                       int(y*(2/chunks+overlap)),
                                       int(x*(2/chunks+overlap))
                                       )
            window_shape_focus = window_shape_morphology[1:]+(1,)
            #
            view_morphology = np.lib.stride_tricks.sliding_window_view(morphology_section, window_shape_morphology)
            view_morphology = view_morphology[:, 
                                              ::view_morphology.shape[1]-1,
                                              ::view_morphology.shape[2]-1,
                                              ...]
            view_morphology = np.reshape(view_morphology,
                                         (z, chunks)+window_shape_morphology[1:]
                                         )
            #
            view_focus = np.lib.stride_tricks.sliding_window_view(focus_section,
                                                                  window_shape_focus)
            view_focus = view_focus[::view_focus.shape[1]-1,
                                    ::view_focus.shape[2]-1,
                                    ...
                                    ]
            view_focus = np.reshape(view_focus,
                                    (chunks,)+window_shape_focus[:-1]+(4,)
                                    )
            #
            for chunk in range(chunks):
                q_m = view_morphology.copy()[:, chunk, ...]
                q_f = view_focus.copy()[chunk, ...]
                if q.ndim != 3:
                    q = np.squeeze(q)
                    if q.ndim != 3:
                        print('something is weird')
                        break
                write_tif(q_m,
                          section,
                          chunk=chunk,
                          resolution=window_shape_morphology[1:])
                write_tif(q_f,
                          section,
                          chunk=chunk,
                          resolution=window_shape_focus[:-1],
                          focus=True
                          )
                for layer in range(z):
                    write_tif(q_m[layer, ...],
                              section,
                              chunk=chunk,
                              layer=layer,
                              resolution=window_shape_morphology[1:],
                              ome=False
                              )
