from pathlib import Path
import argparse
import tomllib
import pickle

import tifffile as tf
import pandas as pd
import numpy as np

# for pca
from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.morphology import label
from aicsimageio.aics_image import imread, AICSImage
from aicsimageio.readers import ome_tiff_reader, tiff_reader, array_like_reader
from aicsimageio.writers import ome_tiff_writer

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
    CS_BENCH = evaluation['CS-BENCH']

    mask_path = '<>'

    #gt = tf.imread(gt_path)
    #mask = tf.imread(mask_path)
    focus_path = Path(f'{home}/{sample}/processed/{section}/morphology/focus/')
    test_path = Path('/data/cephfs-2/unmirrored/groups/sawitzki/Juno/data/2D_CODEX.ome.tiff')

    outdir = Path(f'{home}/{sample}/results/{method}/evaluation/{section}/')
    outdir.mkdir(parents=True, exist_ok=True)

    if PCA:
        with open('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/10xSegEval/eval/pca.pickle', 'rb') as pkl:
            PCA = pickle.load(pkl)

        img = tf.imread(focus_path / 'focus.ome.tif')
        print(img.shape)
        img = np.moveaxis(img, -1, 0)
        print(img.shape)
        writer=ome_tiff_writer.OmeTiffWriter()
        channel_names = [
            'DAPI',
            'ATP1A1_E-Cadherin_CD45',
            '18S_rRNA',
            'alphaSMA_Vimentin'
        ]

        stats = {
            'dim_order': 'CYX',
            'channel_names': channel_names,
            'image_name':'focus',
            'pixel_physical_size': 0.2125,
            'channel_colours': ['red', 'green', 'blue', 'yellow']
        }

        writer.save(
            img,
            uri=Path(focus_path / 'aics.ome.tif'),
            **stats
        )

        img = AICSImage(focus_path / 'aics.ome.tif', reader=ome_tiff_reader.OmeTiffReader)
        print(img)
        #print(img.data)
        print(img.metadata)

        mask = AICSImage(f'{home}/{sample}/results/mesmer/output/{section}/prediction_mem.tif', reader=tiff_reader.TiffReader)
        print(mask.shape)
        #mask = find_boundaries(mask, connectivity=1, mode='inner')
        #print(type(mask))

        print(single_method_eval(img, mask, PCA_model=PCA,
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

    if CS_BENCH:
        gt = label(gt)
        mask = label(mask)

        pm = Metrics(method, outdir=outdir)

        object_metrics = pm.calc_object_stats(gt, mask)

        results = pd.DataFrame(data=object_metrics)

        results.to_csv(outdir / 'CS-BENCH.csv', index=False)
    
    if PD:
        # nothing
        print('nothing')


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
