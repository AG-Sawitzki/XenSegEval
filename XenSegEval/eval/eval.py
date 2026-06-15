import sys
import gzip
from pathlib import Path
import argparse
import pickle

import tifffile as tf
import pandas as pd
import numpy as np
import tomlkit
import geopandas as gpd


# for pca
from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.morphology import label
#from aicsimageio.aics_image import imread, AICSImage
#from aicsimageio.readers import ome_tiff_reader, tiff_reader, array_like_reader
#from aicsimageio.writers import ome_tiff_writer

# for jaccard
from XenSegEval.eval.unet4nuclei.evaluation import (
    compute_af1_results,
    get_false_negatives,
    get_splits_and_merges
)

# for cs-bench
from XenSegEval.eval.cs_benchmark.metrics import Metrics

# 
from XenSegEval.utils import polygon_to_mask, get_config_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image Processing.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    args = parser.parse_args()
    
    method = args.Method
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    section = 'newmem'

    gt = tf.imread(gt_path)

    if method is 'proseg':
        file = 'cell-polygons_layers.geojson.gz'
        polygon_path = Path(
            f'{results}/{method}/output/{section}/{file}'
        )
        shape = gt.shape
        with gzip.open(polygon_path) as file:
            gdf = gpd.read_file(file)
            layers = max(gdf['layer'])
        for layer in range(layer+1):
            mask = polygon_to_mask(polygon_path, shape, layer)
            tf.imwrite(
                f'{results}/{method}/output/{section}/perdiction_l{layer}.tif',
                mask
            )
        
    mask_path = f'{results}/{method}/output/{section}/'

    for file in Path(mask_path).glob('prediction*.tif'):
        dir_name = file.stem.replace('prediction', '')
        if dir_name is not '':
            outdir = Path(
                f'{home}/{sample_name}/results/{method}/evaluation/{section}/{dir_name}'
            )
        else:
            outdir = Path(
                f'{home}/{sample_name}/results/{method}/evaluation/{section}
            )
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
            print('jaccard')
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
            
            split_merges = get_splits_and_merges(
                gt, 
                mask, 
                split_merges, 
                method
            )

            results.to_csv(outdir / 'results.csv', index=False)
            false_negative.to_csv(outdir / 'false_negative.csv', index=False)
            split_merges.to_csv(outdir / 'split_merges.csv', index=False)

        if CS_BENCH:
            print('cs_bench')
            gt = label(gt)
            mask = label(mask)

            pm = Metrics(method, outdir=outdir)

            object_metrics = pm.calc_object_stats(gt, mask)

            results = pd.DataFrame(data=object_metrics)

            results.to_csv(outdir / 'CS-BENCH.csv', index=False)
        
        if PD:
            # nothing
            print('nothing')
