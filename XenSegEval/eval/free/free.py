from XenSegEval.utils import get_config_args
# from XenSegEval.eval.utils import (
#     prepare_ProSeg,
#     polygon_to_mask,
# )

import os
import sys
import gzip
import json
from pathlib import Path
import argparse
import pickle

import tifffile
import pandas as pd
import numpy as np
import tomlkit
# import geopandas as gpd

from typing import Any, Union
from pathlib import PosixPath


# for pca
from CellSegmentationEvaluator.single_method_eval import (
    single_method_eval
)
from CellSegmentationEvaluator.single_method_eval_3D import (
    single_method_eval_3D
)
# from skimage.segmentation import find_boundaries, relabel_sequential
# from skimage.morphology import label
from aicsimageio.aics_image import imread, AICSImage
from aicsimageio.readers import (
    ome_tiff_reader, tiff_reader, array_like_reader
)
from aicsimageio.writers import OmeTiffWriter


def read_and_eval_seg(
    img_path: Union[str, os.PathLike, PosixPath],
    mask_path: Union[str, os.PathLike, PosixPath],
    outdir: Union[str, os.PathLike, PosixPath],
    pixelsizes: tuple,
    PCA_model: str = '2Dv1.5',
) -> dict:
    '''Wrapper for CSE. Prepares input.

    Parameters
    ----------
    img_path :  Path
        Path to immuno-histology image. Here, focus.tif
    mask_path : Path
        Path to the Algorithms output.
    outdir : Path, optional
        Path to the output directory.
    pixelsize : tuple
        Contains the sizes of a voxel in x,y,z direction.
    PCA_model :  string, optional
        name of the model to use. Default 2Dv1.5

    Returns
    ----------
        out : dict
            Dictionary of Segmentation Evaluation Metrics and QualityScore.
        Addionally saves the dictionary as json under `output_direcory`.
    '''
    outdir = Path(outdir)
    aimg = AICSImage(img_path)
    physical_size_x, physical_size_y, physical_size_z = pixelsizes
    # print(xmltodict.parse(aimg.metadata.to_xml()))
    # physical_size_x, physical_size_y, physical_size_z=extract_voxel_size_from_tiff(img_path)
    # print(physical_size_x, physical_size_y, physical_size_z)
    img = {}
    iheadtail = os.path.split(img_path)
    img["path"] = iheadtail[0]
    img["name"] = iheadtail[1]
    img["img"] = aimg
    img["data"] = aimg.get_image_data()
    img["pixelsizes"] = (
        physical_size_x,
        physical_size_y,
        physical_size_z
    )
    # print(img["data"].shape)

    amask = AICSImage(mask_path)
    mask = {}
    mheadtail = os.path.split(mask_path)
    mask["path"] = mheadtail[0]
    mask["name"] = mheadtail[1]
    mask["img"] = amask
    mask["data"] = amask.get_image_data()
    # print(mask["data"].shape)

    if not physical_size_x:
        print('File missing physical pixel sizes in XY.')
        # physical_size_x = float(input("Enter size of x pixels: "))
        # physical_size_y = float(input("Enter size of y pixels: "))

    if img["data"].shape[2] == 1:
        seg_metrics = single_method_eval(
            img, mask, PCA_model, outdir,
            0, 0, physical_size_x, physical_size_y
        )
    else:
        if not physical_size_z:
            print('File missing physical pixel size in Z.')
            # physical_size_z = float(input("Enter size of z pixels: "))
        seg_metrics = single_method_eval_3D(
            img, mask, PCA_model, outdir,
            'um', physical_size_x, physical_size_y, physical_size_z
        )

    struct = {"Segmentation Evaluation Metrics v1.5": seg_metrics}
    if outdir is not None:
        with open(
                outdir / (img["name"] + "-seg_eval.json"), "w"
        ) as json_file:
            json.dump(struct, json_file)

    return seg_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Eval.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    parser.add_argument('-s', '--Section', help='Section segmented.')
    args = parser.parse_args()

    method = args.Method
    section = args.Section
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    focus_path = Path(f'{processed}/{section}/morphology/focus/')


    mask_path = f'{results}/{method}/output/{section}/'

    files = sorted(Path(mask_path).glob('prediction*.tif'))
    if method in ['segger', 'xenium']:
        masks = [tifffle.imread(mask) for mask in files]
        masks = [np.array([masks])]
    else:
        masks = [tifffile.imread(mask) for mask in files]

    #if PCA['use'] and method in PCA_CAPABLE:
    # with open(
    #     './XenSegEval/eval/pca.pickle', 'rb'
    # ) as pkl:
    #     pca = pickle.load(pkl)

    img = tifffile.imread(focus_path / 'focus.ome.tif')
    print(img.shape)
    img = np.moveaxis(img, -1, 0)
    print(img.shape)
    writer = OmeTiffWriter()
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
        'pixel_physical_size': {
            'Z': imagestats['pixelsize_z'],
            'Y': imagestats['pixelsize_xy'],
            'X': imagestats['pixelsize_xy']
        },
        'channel_colours': ['red', 'green', 'blue', 'yellow']
    }

    writer.save(
        img.astype(np.int32),
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

    for i, mask in enumerate(masks):
        outdir = mask_path.parents[1] / 'evaluation'
        outdir.mkdir(parents=True, exist_ok=True)

        mask = np.squeeze(mask, )
        mask = mask.astype(np.int32)
        # mask = AICSImage(
        #     file,
        #     reader=tiff_reader.TiffReader
        # )
        new_file = Path(files[i].parent / f'aics_{files[i].name}')
        writer.save(
            mask,
            uri=new_file,
            dim_order='CYX',
            image_name=f'{method}_{file.stem}',
            # channel_names=[
            #     'cell',
            #     'nucleus'
            # ],
            pixel_physical_size=imagestats['pixelsize_xy']
        )
        print(mask.shape)
        # mask = find_boundaries(mask, connectivity=1, mode='inner')
        # print(type(mask))

        # output = single_method_eval(
        #     AICSImage(focus_path / 'aics.ome.tif'),
        #     AICSImage(file.parent / f'aics_{file.name}'),
        #     PCA_model='2Dv1.5',  # pca,
        #     output_dir=(
        #         '/data/cephfs-2/unmirrored/groups/'
        #         'sawitzki/Juno/eval-test/PCA'
        #     )
        # )
        output = read_and_eval_seg(
            focus_path / 'aics.ome.tif',
            new_file,
            PCA_model='2Dv1.5',
            outdir=(
                f'{outdir}'
            ),
            pixelsizes=(
                imagestats['pixelsize_xy'],
                imagestats['pixelsize_xy'],
                imagestats['pixelsize_z']
            )
        )

        # if PD['use']:
        #     # nothing
        #     print('nothing')
        #     break
        #     if not os.path.exists(results_dir):
        #         os.makedirs(results_dir)
        #     pmask_save_dir = os.path.join(results_dir, "pmasks")
        #     if not os.path.exists(pmask_save_dir):
        #         os.makedirs(pmask_save_dir)
        #     write_pmasks(sample_ids, radii, methods, datapath, pmask_save_dir)
        #     filtered_pmask_save_dir = os.path.join(results_dir, "filtered_pmasks")
        #     if not os.path.exists(filtered_pmask_save_dir):
        #         os.makedirs(filtered_pmask_save_dir)
        #     filter_pmasks(sample_ids, pmask_save_dir, filtered_pmask_save_dir, num_agree, methods)
        #     precision, recall = evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, num_agree, len(methods))
        #     plot_precision_recall(precision, recall, methods, sample_ids)
        #     for method in methods:
        #         pmask_save_dir = os.path.join(results_dir, f"pmasks_{method}_out")
        #         if not os.path.exists(pmask_save_dir):
        #             os.makedirs(pmask_save_dir)
        #         methods_ = [m for m in methods if m != method]
        #         write_pmasks(sample_ids, radii, methods_, datapath, pmask_save_dir)
        #     for method in methods:
        #         pmask_save_dir = os.path.join(results_dir, f"pmasks_{method}_out")
        #         filtered_pmask_save_dir = os.path.join(results_dir, f"filtered_pmasks_{method}_out")
        #         if not os.path.exists(filtered_pmask_save_dir):
        #             os.makedirs(filtered_pmask_save_dir)
        #         methods_ = [m for m in methods if m != method]
        #         filter_pmasks(sample_ids, pmask_save_dir, filtered_pmask_save_dir, num_agree, methods_)
        #     ablation_eval = {}
        #     for method in methods:
        #         filtered_pmask_save_dir = os.path.join(results_dir, f"filtered_pmasks_{method}_out")
        #         precision, recall = evaluate_masks(sample_ids, filtered_pmask_save_dir, radii, num_agree, len(methods) - 1)
        #         ablation_eval[f'{method}_out'] = (precision, recall)
