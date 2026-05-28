from pathlib import Path

import tifffile as tf
import pandas as pd
import numpy as np

# for pca
from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.morphology import label
from aicsimageio.aics_image import imread

# for jaccard
from unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)

# for cs-bench
from cs_benchmark.metrics import Metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    parser.add_argument('-s', '--Section', help='Section to run the evaluation on.')
    args = parser.parse_args()
    
    method = args.Method
    section = args.Section
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    paths = config['paths']
    imagestats = config['ImageStats']
    evaluation = config['evaluation']

    # define paths
    home = paths['home']
    sample = paths['sample_name']
    data = paths['data_path']
    gt_path = paths['ground_truth']

    #
    PD = evaluation['PD']
    PCA = evaluation['PCA']
    JACCARD = evaluation['JACCARD']
    CS-BENCH = evaluation['CS-BENCH']

    mask_path = '<>'

    gt = tf.imread(gt_path)
    mask = tf.imread(mask_path)

    outdir = Path(f'{sample}/results/{method}/evalueation/{section}/')

    if PCA:
        img = imread('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/TMA3/processed/9/morphology/focus/focus.ome.tif')


        mask = np.load('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/results/res_mesmer/36_segmentation_predictions_nuc_dapi-mem.npy')
        print(mask.shape)
        mask = find_boundaries(mask, connectivity=1, mode='inner')
        print(type(mask))

        print(single_method_eval(img, mask,
            output_dir='/data/cephfs-2/unmirrored/groups/sawitzki/Juno/eval-test/PCA'
            )
        )

    if JACCARD:
        results = pd.DataFrame(
            columns=["Method", "Threshold", "F1", "Jaccard", "TP", "FP", "FN"]
        )
        false_negatives = pd.DataFrame(
            columns=["False_Negative", "Area"]
        )
        split_merges = pd.DataFrame(
            columns=["Method", "Merges", "Splits"]
        )

        gt = label(gt)

        if len(gt.shape) == 3:
            gt = gt[:,:,0]
        
        mask = tf.imread(mask_path)

        gt = relabel_sequential(gt)[0]
        mask = relabel_sequential(mask)[0]

        results = compute_af1_results(
            gt, 
            mask, 
            results, 
            method
        )
        
        false_negatives = get_false_negatives(
            gt, 
            mask, 
            false_negatives, 
            method
        )
        
        splits_merges = get_splits_and_merges(
            gt, 
            mask, 
            splits_merges, 
            method
        )

        results.to_csv(outdir / 'results.csv', index=False)
        false_negative.to_csv(outdir / 'false_negative.csv', index=False)
        split_merges.to_csv(outdir / 'split_merges.csv', index=False)

    if CS-BENCH:
        gt = label(gt)
        mask = label(mask)

        pm = Metrics(method, outdir=outdir)

        object_metrics = pm.calc_object_stats(gt, mask)

        results = pd.DataFrame(data=object_metrics)

        results.to_csv(outdir / 'CS-BENCH.csv', index=False)
    
    if PD:


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
