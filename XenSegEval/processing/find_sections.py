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
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import zarr
import cv2

# types
from numpy.typing import ArrayLike
from typing import Any, Union


def get_weighted_distance(
    centre: Union[tuple, list],
    weightx: float = 0.25,
    weighty: float = 1
) -> float:
    """Get weighted distance of an area's centre from [0,0].

    Parameters:
    ----------
        centre : tuple or list
            Centre of the area. Given in (y,x).
        weightx : float
            How large the impact of x is on the distance.
            lower x = similar y values have lower distance.
        weighty : float
            How large the impact of x,y is on the distance.
            lower y = similar x values have lower distance.

    Retruns
    ----------
        out : float
            Distance as float.
    """
    x, y = centre
    return np.sqrt((x*weightx)**2 + (y*weighty)**2)


def find_rois(
    shape_org: tuple,
    image_subres: ArrayLike,
    n_roi: int
) -> Union[list, ArrayLike]:
    """Sort the contours by area.

    Parameters:
    ----------
        shape_org : tuple
            Max resolution of img.
        image_subres : ArrayLike
            Lowest subresolution of image.
        n_roi : int
            Expected # of regions of interest.
            Should be equivalent to the number of tissue-samples on the slide.

    Returns
    ----------
        out : ArrayLike or list
            Contours of significant size.
    """
    z, y, x = shape_org

    subres_centre = np.uint8(image_subres[z//2])

    subres_dilated = cv2.dilate(
        subres_centre,
        np.ones((3, 3)),
        iterations=5
    )
    _, subres_binary = cv2.threshold(
        subres_dilated,
        127, 255, 0
    )

    contours, _ = cv2.findContours(
        subres_binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # keep contours with significant size
    values = []
    dtype = [('area', float), ('wd', float), ('y', float), ('x', float)]

    for c in contours:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        wd = get_weighted_distance([x, y])
        values.append((w*h, wd, y, x))

    values_arr = np.array(values, dtype=dtype)
    # sort by area and find smallest allowed roi
    values_arr_sorted = np.sort(values_arr, kind='stable', order='area')
    smallest_allowed_roi = values_arr_sorted['area'][-n_roi]
    # sort by weighted_distance
    values_arr_wd_args = np.argsort(values_arr, kind='stable', order='wd')
    values_arr_wd_sorted = np.sort(values_arr, kind='stabe', order='wd')
    contours_wd_sorted = [contours[index] for index in values_arr_wd_args]

    mask = values_arr_wd_sorted['area'] >= smallest_allowed_roi
    nroi_contours = [
        contours_wd_sorted[index] for index, boolean in enumerate(mask)
        if boolean
    ]

    return nroi_contours, subres_centre


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ROIs')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'ROIs')
    globals().update(variables)

    # load morpho and focus:
    print('to load')
    morphology_store = imread(f'{data_path}/morphology.ome.tif', aszarr=True)
    morphology_zarr = zarr.open(morphology_store, mode='r')

    subres_lvls = [lvl for lvl in morphology_zarr]
    subres_max = max(subres_lvls)
    subres_min = min(subres_lvls)

    morphology_org = morphology_zarr[subres_min]
    shape_org = morphology_org.shape()

    print('Searching for ROIs...')
    sections_dict = {}

    morphology_subres = morphology_zarr[subres_max]

    roi_list, subres_centre = find_rois(
        shape_org, morphology_subres,
        preprocessing['n_roi']
    )

    with tqdm(
        total=len(roi_list),
        desc='Saving Coordinates',
        ncols=79,
        leave=True
    ) as search_bar:
        buffer = preprocessing['buffer']

        z, y, x = morphology_org.shape
        z_, y_, x_ = morphology_subres.shape

        rf_x = int(x/x_)
        rf_y = int(y/y_)

        for section, contour in enumerate(roi_list):
            # add roi to scaled image to check for regions
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                subres_centre, (x, y), (x+w, y+h),
                (255, 255, 255), 2
            )
            cv2.putText(
                img=subres_centre,
                text=str(section),
                org=(int(x+w/4), int(y+h/2)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2,
                color=(255, 255, 255),
                thickness=2,
                bottomLeftOrigin=False
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
            memory_percentage = get_memory_usage_percentage()
            search_bar.set_description(
                f'Saving Coordinates | %MEM: {memory_percentage:.2f}'
            )
            search_bar.update(1)

        with open(processed / 'sections_px.json', 'w') as f:
            json.dump(sections_dict, f)

        imwrite(
            processed / 'marked_regions-of-interest.tif',
            subres_centre
        )
