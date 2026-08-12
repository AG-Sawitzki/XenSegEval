from XenSegEval.processing.utils import (
    process_roi
)

from pathlib import Path
import argparse

import tifffile
import tomlkit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='boundaries')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-m', '--Method',
        help='Method to convert from polygon(.geojson) to mask(.tif).'
    )

    args = parser.parse_args()

    config_path = args.Config
    method = args.Method

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'boundaries')
    globals().update(variables)

    path = Path(f'{results}/{method}/')
    files = list(path.rglob('*.tif'))

    processed = Path(f'{results}').parent / 'processed'
    _section_ = list(sections)[0]
    file = Path(f'{processed}/{_section_}/morphology/focus/focus.ome.tif')
    img = tifffile.imread(file)
    shape = img.shape[:2]

    for file in files:
        name = method + file.stem[file.rfind('_'):]
        if method == mesmer:
            for i, mode in enumerate(['cell', 'nucleus']):
                img = tifffile.imread(file)[i, ...]
                name = '_'.join(name, mode)
                process_roi(
                    img,
                    file.parent / name,
                )
        else:
            process_roi(
                img,
                file.parent / name,
            )

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