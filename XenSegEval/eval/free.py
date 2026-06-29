from XenSegEval.utils import get_config_args
from XenSegEval.eval.utils import (
    prepare_ProSeg,
    polygon_to_mask,
)

import sys
import gzip
from pathlib import Path
import argparse
import pickle

import tifffile
import pandas as pd
import numpy as np
import tomlkit
import geopandas as gpd


# for pca
from CellSegmentationEvaluator.single_method_eval import single_method_eval
# from skimage.segmentation import find_boundaries, relabel_sequential
# from skimage.morphology import label
from aicsimageio.aics_image import imread, AICSImage
from aicsimageio.readers import (
    ome_tiff_reader, tiff_reader, array_like_reader
)
from aicsimageio.writers import ome_tiff_writer

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

    gt = tifffile.imread(gt_path)

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
        mask = tifffile.imread(file)

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

        if PCA['use'] and method in PCA_CAPABLE:
            with open(
                '/data/cephfs-1/work/groups/sawitzki/'
                'users/juno12_c/XenSegEval/eval/pca.pickle', 'rb'
            ) as pkl:
                PCA = pickle.load(pkl)

            img = tifffile.imread(focus_path / 'focus.ome.tif')
            print(img.shape)
            img = np.moveaxis(img, -1, 0)
            print(img.shape)
            writer = ome_tiff_writer.OmeTiffWriter()
            channel_names = [
                'DAPI',
                'ATP1A1_E-Cadherin_CD45',
                '18S_rRNA',
                'alphaSMA_Vimentin'
            ]

            stats = {
                'dim_order': 'CYX',
                'channel_names': channel_names,
                'image_name': 'focus',
                'pixel_physical_size': 0.2125,
                'channel_colours': ['red', 'green', 'blue', 'yellow']
            }

            writer.save(
                img,
                uri=Path(focus_path / 'aics.ome.tif'),
                **stats
            )

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

        if PD['use']:
            # nothing
            print('nothing')
