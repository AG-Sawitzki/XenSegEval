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
    except AttributeError:
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
    gdf = gpd.GeoDataFrame(data=data)
    gdf.set_index(['layer', 'name'])
    print(' - Saving GeoDataFrame')
    gdf.to_file(npy_base_output_path, driver='GeoJSON', index=True)


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

    preprocessing = config['preprocessing']
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


# use this for saving an overview image for the segmentation
# and possibly as source for 
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# from matplotlib.font_manager import FontProperties
# from skimage.segmentation import find_boundaries
# import matplotlib.pyplot as plt
# from tifffile import imread
# import geopandas as gpd
# import numpy as np

# from pathlib import Path
# data = Path('/data/cephfs-2/unmirrored/groups/sawitzki/Juno')
# res = Path('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/results/')


# morphology_focus = imread(data / 'data/data_processed/image-data_processed/morphology/36/focus/morphology_focus_0002_36.ome.tif')
# morphology_focus = np.moveaxis(morphology_focus, 0, -1)
# print(morphology_focus.shape)

# morphology_focus_normalized = (morphology_focus[...,:3]-morphology_focus[...,:3].min())/(morphology_focus[...,:3].max()-morphology_focus[...,:3].min())
# morphology_focus_normalized_fov = morphology_focus_normalized[:6040,:6431,:]

# print(morphology_focus_normalized_fov.shape)

# ## for proseg
# #vector = gpd.read_file(res / 'res_proseg/cell-polygons_36_relative.geojson')
# #vector_scaled = vector.scale(1/0.2125, 1/0.2125, origin = (0,0))

# ### for cv++ ## needs H&E - which i don't want to plot rn
# ##vector_scaled = gpd.read_file(res / 'res_plusplus/Xenium_ID_22391_x-7109_y-7754_w-8900_h-9501_vips/cells.geojson')

# ## for cpsam
# #boundary = np.load(res / 'res_cellpose_sam/morphology_L5-8_q00.ome_seg.npy', allow_pickle = True).item()['outlines'][2]
# #print(boundary.shape)

# ## for stardist
# #masks = np.load(res / 'res_stardist/36_q00.npy')
# #boundary = find_boundaries(masks, connectivity = 1, mode = 'inner')

# ## for dissect
# #masks = np.load(res / 'res_dissect/mask.npy')
# #boundary = find_boundaries(masks, connectivity = 1, mode = 'inner')
# #boundary = boundary[:6040,:6431]

# ## for mesmer
# #segmentation_predictions_nuc = np.load(res / 'res_mesmer/36_segmentation_predictions_nuc_dapi-none.npy')
# #boundary = find_boundaries(segmentation_predictions_nuc[0, :6040,:6431, 0], connectivity=1, mode='inner')

# #morphology_focus_normalized_fov[boundary > 0, :] = 1
# #print(morphology_focus_normalized_fov.shape)

# fz = 48
# dimy, dimx = 6040, 6431 # 10983, 11694

# fig, ax = plt.subplots()
# fig.set_frameon(False)
# fig.set_size_inches(dimy/100, dimx/100)
# ax.tick_params(axis='both', which='major', labelsize=fz)
# plt.xlabel('x_location in px', fontsize=fz)
# plt.ylabel('y_location in px', fontsize=fz)
# ax.set_aspect('equal','box')

# asb = AnchoredSizeBar(ax.transData,
#                 size = 941.1764705882354,
#                 label = '200 Âµm',
#                 loc = 'lower left',
#                 frameon = False,
#                 size_vertical = 47.05882352941177,
#                 color = 'white',
#                 fontproperties = FontProperties(size = fz))

# ax.add_artist(asb)

# ax.set_xlim(0,dimx)#11694)
# ax.set_ylim(0,dimy)#10983)

# print('set style. plotting...')

# ax.imshow(morphology_focus_normalized_fov)#, cmap='gray')#, interpolation='nearest')

# #vector_scaled.boundary.plot(ax = ax, aspect = 'equal', color = 'white')

# print('saving image.')
# fig.savefig(res / 'res_mesmer/36_q00_outlines_mesmer_dapi-none.png', dpi=100, bbox_inches='tight', pad_inches=0)