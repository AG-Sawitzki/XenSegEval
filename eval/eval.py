from pathlib import Path

from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries
from aicsimageio.aics_image import imread
import tifffile as tf
import numpy as np


if PCA:
    img = imread('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/TMA3/processed/9/morphology/focus/focus.ome.tif')


    mask = np.load('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/results/res_mesmer/36_segmentation_predictions_nuc_dapi-mem.npy')
    mask = find_boundaries(mask, connectivity=1, mode='inner')
    print(type(mask))

    print(single_method_eval(img, mask,
        output_dir='/data/cephfs-2/unmirrored/groups/sawitzki/Juno/eval-test/PCA'
        )
    )

# import cv2
# import geopandas as gpd

# gdf = gpd.read_file('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/labels/roi.geojson')
# xx, yy = gdf.iloc[0]['geometry'].exterior.xy

# x = np.array(xx)
# y = np.array(yy)

# x.shape = (len(x),1)
# y.shape = (len(y),1)

# xy = np.hstack((x,y))

# mm = cv2.moments(xy)

# source = tifffile.imread('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/labels/13-membrane-reordered-overlap_labels.tif')
# template = tifffile.imread('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/labels/13-membrane-reordered-no-overlap_labels.tif')

# contours_src, _ = cv2.findContours(np.uint8(source), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours_tmp, _2 = cv2.findContours(np.uint8(template), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# print(len(contours_src))
# print(len(contours_tmp))

# for i, c in enumerate(contours_src):
#     similarity = cv2.matchShapes(c, contours_tmp[i], 1, 0.0)
#     print(similarity)
