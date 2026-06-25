# for jaccard
# from XenSegEval.eval.unet4nuclei.evaluation import (
#     compute_af1_results,
#     get_false_negatives,
#     get_splits_and_merges
# )
# for cs-bench
# from XenSegEval.eval.cs_benchmark.metrics import Metrics
# Utils
from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    prepare_ProSeg,
    polygon_to_mask,
    wrapper_cs,
    wrapper_u4n
)

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
# from CellSegmentationEvaluator.single_method_eval import single_method_eval
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.morphology import label
# from aicsimageio.aics_image import imread, AICSImage
# from aicsimageio.readers import (
#     ome_tiff_reader, tiff_reader, array_like_reader
# )
# from aicsimageio.writers import ome_tiff_writer

PCA_CAPABLE = [
    'cpsam',
    'dinocell',
    'dissect',
    'mesmer',
    'proseg',
    'stardist'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Eval.')
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

    focus_path = Path(f'{processed}/{section}/morphology/focus/')

    gt = tf.imread(gt_path)

    # prepare Proseg output
    if method == 'proseg':
        files = [
            'cell-polygons.geojson.gz',
            'cell-polygons_layers.geojson.gz'
        ]
        for file in files:
            polygons = Path(
                f'{results}/{method}/output/{section}/{file}'
            )
            output_path = Path(
                f'{results}/{method}/outupt/{section}'
            )
            shape = gt.shape
            prepare_ProSeg(polygons, output_path, shape)

    mask_path = f'{results}/{method}/output/{section}/'

    for file in Path(mask_path).glob('prediction*.tif'):
        print(file)
        mask = tf.imread(file)

        dir_name = file.stem.replace('prediction', '')

        if dir_name != '':
            outdir = Path(
                f'{home}/{sample_name}/results/{method}/'
                f'evaluation/{section}/{dir_name}'
            )
        else:
            outdir = Path(
                f'{home}/{sample_name}/results/{method}/evaluation/{section}'
            )

        outdir.mkdir(parents=True, exist_ok=True)

        if PCA and method in PCA_CAPABLE:
            with open(
                '/data/cephfs-1/work/groups/sawitzki/'
                'users/juno12_c/XenSegEval/eval/pca.pickle', 'rb'
            ) as pkl:
                PCA = pickle.load(pkl)

            # img = tf.imread(focus_path / 'focus.ome.tif')
            # print(img.shape)
            # img = np.moveaxis(img, -1, 0)
            # print(img.shape)
            # writer = ome_tiff_writer.OmeTiffWriter()
            # channel_names = [
            #     'DAPI',
            #     'ATP1A1_E-Cadherin_CD45',
            #     '18S_rRNA',
            #     'alphaSMA_Vimentin'
            # ]

            # stats = {
            #     'dim_order': 'CYX',
            #     'channel_names': channel_names,
            #     'image_name': 'focus',
            #     'pixel_physical_size': 0.2125,
            #     'channel_colours': ['red', 'green', 'blue', 'yellow']
            # }

            # writer.save(
            #     img,
            #     uri=Path(focus_path / 'aics.ome.tif'),
            #     **stats
            # )

            img = AICSImage(
                focus_path / 'aics.ome.tif',
                reader=ome_tiff_reader.OmeTiffReader
            )
            print(img)
            # print(img.data)
            print(img.metadata)

            mask = AICSImage(
                f'{home}/{sample}/results/mesmer/'
                f'output/{section}/prediction_mem.tif',
                reader=tiff_reader.TiffReader
            )
            print(mask.shape)
            # mask = find_boundaries(mask, connectivity=1, mode='inner')
            # print(type(mask))

            print(
                single_method_eval(
                    img, mask, PCA_model=PCA,
                    output_dir='/data/cephfs-2/unmirrored/groups/'
                               'sawitzki/Juno/eval-test/PCA'
                )
            )

        if JACCARD:
            print('jaccard')
            # if method == 'mesmer':
            #     mask = mask.squeeze()[0, ...]

            # mask_l = label(mask)
            # gt_l = label(gt)

            # mask_rl = relabel_sequential(mask_l)[0]
            # gt_rl = relabel_sequential(gt_l)[0]

            # results = pd.DataFrame(
            #     columns=[
            #         'Method', 'Threshold', 'F1',
            #         'Jaccard', 'TP', 'FP', 'FN'
            #     ]
            # )
            # false_negatives = pd.DataFrame(
            #     columns=['False_Negative', 'Area']
            # )
            # split_merges = pd.DataFrame(
            #     columns=['Method', 'Merges', 'Splits']
            # )

            # results = compute_af1_results(
            #     gt_rl,
            #     mask_rl,
            #     results,
            #     method
            # )

            # false_negatives = get_false_negatives(
            #     gt_rl,
            #     mask_rl,
            #     false_negatives,
            #     method
            # )

            # split_merges = get_splits_and_merges(
            #     gt_rl,
            #     mask_rl,
            #     split_merges,
            #     method
            # )
            results, false_negatives, split_merges = wrapper_u4n(
                mask,
                gt,
                method=method
            )
            results.to_csv(outdir / 'results.csv', index=False)
            false_negatives.to_csv(outdir / 'false_negatives.csv', index=False)
            split_merges.to_csv(outdir / 'split_merges.csv', index=False)

        if CS_BENCH:
            print('cs_bench')
            # # expand dims. requires 3D (batch, y, x)
            # # or 4D (batch, y, x, chan)
            # gt_x = np.expand_dims(gt, axis=0)
            # mask_x = np.expand_dims(mask, axis=0)

            # pm = Metrics(method, outdir=outdir)

            # object_metrics = pm.calc_object_stats(gt_x, mask_x)

            # results = pd.DataFrame(data=object_metrics)
            results = wrapper_cs(mask, gt, method=method, outdir=outdir)
            results.to_csv(outdir / 'CS-BENCH.csv', index=False)

        if PD:
            # nothing
            print('nothing')
