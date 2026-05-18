import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile
import cv2

from shapely.geometry import Polygon
import geopandas as gpd
import json


from pathlib import Path
import configparser
import argparse
import tomllib

# function form cellpose.utils
def outlines_list(masks, multiprocessing_threshold=1000, multiprocessing=None):
    """
    Get outlines of masks as a list to loop over for plotting.

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
    """
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
    """
    Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    """
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
    """
    Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.
    """
    if num_processes is None:
        num_processes = cpu_count()
    unique_masks = np.unique(masks)[1:]
    with Pool(processes=num_processes) as pool:
        outpix = pool.map(get_outline_multi, [(masks, n) for n in unique_masks])
    return outpix

# function form cellpose.utils
def get_outline_multi(args):
    """
    Get the outline of a specific mask in a multi-mask image.

    Args:
        args (tuple): A tuple containing the masks and the mask number.

    Returns:
        numpy.ndarray: The outline of the specified mask as an array of coordinates.

    """
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
def process_roi(npy_data, npy_base_output_path):
    """
    Get the polgyons from the prediction-masks using cellpose.utils functions
    Saves them as a GeoDataFrame (geojson)
    
    Args:
        npy_data: The numpy.ndarray of the masks.
        npy_base_output_path: Path to save the geojson.
    
    Returns:
        Nothing. Automatically saves the GDF.
    """
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
    gdf.to_file(npy_base_output_path, driver='GeoJSON', index = True)


if __name__ == '__main__':

    # define paths
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-l', '--Labels', help='Path to label to convert to polygons.')
    args = parser.parse_args()

    config_path = args.Config
    labels_path = args.Labels

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    preprocessing = config['precossing']
    paths = config['paths']
    imagestats = config['ImageStats']

    home = paths['home']
    
    results = Path(f'{home}/{sample}/results')
    results.mkdir(parents=True, exist_ok=True)

    if labels_path:
        mask = tifffile.imread(labels_path)
        process_roi(mask, labels_path+'.geojson')
    else:
        for method in sections[sections.index('METHODS')+1:]:
            input_path = Path(results / '{method}/output/mask.npy')
            # cpsam has items
            # mesmer needs to be squeezed
            # dissect wrong shape?? 10984, 10985 not 10983, 11694 ?? but! gives 'raw_boxes' !!
            # ucs also wrong shape??? 10758, 11694 not 10983, 11694
            output_path = Path(results / '{method}/polygons/roi.geojson'.format(method=method))
            mask = np.load(input_path, allow_pickle)
            process_roi(mask, output_path)