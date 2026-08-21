from XenSegEval.utils import (
    get_config_args,
    get_memory_usage_percentage
)
from XenSegEval.processing.utils import (
    get_weighted_distance,
    find_rois
)

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
    shape_org = morphology_org.shape

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
            sections_dict[str(section)] = {
                'x': [x_min, x_max],
                'y': [y_min, y_max]
            }
            memory_percentage = get_memory_usage_percentage()
            search_bar.set_description(
                f'Saving Coordinates | %MEM: {memory_percentage:.2f}'
            )
            search_bar.update(1)

        with open(sections_path, 'w') as f:
            json.dump(sections_dict, f)

        imwrite(
            processed / 'marked_regions-of-interest.tif',
            subres_centre
        )
