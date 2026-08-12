from XenSegEval.utils import (
    depth,
    get_section_coords
)

import os
import gzip
import json
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from shapely import (
    wkb,
    affinity,
    Polygon,
    geometry
    # transform
)
import cv2
import tifffile
import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib.pyplot as plt

from typing import Any, Union
from pathlib import PosixPath
from numpy.typing import ArrayLike


TABLE = pa.lib.Table
GDF = gpd.geodataframe.GeoDataFrame
PDF = pl.dataframe.frame.DataFrame
DF = pd.DataFrame

MPG = geometry.multipolygon.MultiPolygon


def check_colour(
    r: int,
    g: int,
    b: int
) -> tuple:
    '''Gives a new rgb colour-tuple, incremented by 1.

    Parameters
    ----------
        r : int
            red
        g : int
            green
        b : int 
            blue

    Returns
    ----------
        out : tuple
        Tuple of (r,g,b)
    '''
    if r < 255:
        r += 1
    else:
        if g < 255:
            g += 1
            r = 0
        else:
            if b < 255:
                b += 1
                g = 0
                r = 0
            else:
                return None
    return (r, g, b)


def polygon_to_mask(
    gdf: GDF,
    shape: tuple,
    layer: int,
) -> ArrayLike:
    '''GeoJson Polygons to masks in a TIF.

    Parameters
    ----------
        gdf : GeoDataFrame
            path to geojson(.gz) or geodataframe.
        shape : tuple
            Shape of the image the Polygons belong to.
        layer : int
            Layer to keep.

    Retruns
    ----------
        out : ArrayLike
            Masks in numpy-array.
    '''
    r, g, b = (0,)*3
    img = np.zeros(shape, np.uint8)
    if type(layer) is int:
        gds = gdf[gdf['layer'] == layer]['geometry']
    else:
        gds = gdf['geometry']
    for pg in gds:
        if isinstance(pg, MPG):
            for lr in pg.geoms:
                pl = np.array(list(lr.exterior.coords))
                cv2.fillPoly(img, np.int32([pl]), (r, g, b))
                r, g, b = check_colour(r, g, b)
        else:
            pl = np.array(list(pg.exterior.coords))
            cv2.fillPoly(img, np.int32([pl]), (r, g, b))
            r, g, b = check_colour(r, g, b)
    return img


def wrap_ptm(
    polygons: Union[str, os.PathLike, PosixPath, GDF],
    output_path: Union[str, os.PathLike, PosixPath],
    shape: tuple,
    mode: str = None,
) -> None:
    '''A wrapper for polygon_to_mask.

    Parameters
    ----------
        polygons : Path
            Path to the geojson file or GeoDataFrame.
        output_path : Path
            Path to the dir to save the masks under.
        shape : tuple
            Shape of the corresponding groundtruth or known area shape.

    Returns
    ----------
        out : None
            Saves masks as `.tif` in output_dir.
    '''
    if Path(polygons).suffix == '.gz':
        with gzip.open(polygons) as file:
            gdf = gpd.read_file(file)
    elif Path(polygons).suffix == '.geojson':
        gdf = gpd.read_file(polygons)
    elif isinstance(gdf, GDF):
        gdf = polygons
    else:
        print('input not path nor GeoDataFrame.')

    try:
        layers = max(gdf['layer'])
        for layer in range(layers+1):
            mask = polygon_to_mask(gdf, shape, layer)
            tifffile.imwrite(
                output_path / f'prediction_l{layer}.tif',
                mask
            )
    except KeyError:
        mask = polygon_to_mask(gdf, shape, layer=None)
        if mode:
            file = output_path / f'prediction_{mode}.tif'
        else:
            file = output_path / 'prediction.tif'
        tifffile.imwrite(
            file,
            mask
        )


def pixelate(
    table: Union[PDF, GDF],
    pixelsize_xy: float,
)-> Union[PDF, GDF]:
    if isinstance(table, PDF):
        columns = [
            c for c in table.columns if (
            f'vertex_' in c or f'_location' in c)
        ]
        pixeled = table.select(
            (pl.col(columns[0]) / pixelsize_xy).round(0),
            (pl.col(columns[1]) / pixelsize_xy).round(0),
        )
        table = table.update(pixeled)
        # print(table)
    if isinstance(table, GDF):
        geometry = table['geometry'].transform(
            lambda x: (x * np.array([
                1/pixelsize_xy, 1/pixelsize_xy
            ])).round(0)
        )
        table['geometry'] = geometry
    return table


def filter_by_location(
    table: Union[PDF, GDF],
    coords,
) -> Union[PDF, GDF]:
    assert depth(coords) == 1, \
        f'`coords` incorrect: {coords}\n Consult example .json!'
    if isinstance(table, PDF):
        for axis, coords in coords.items():
            column = [
                c for c in table.columns if (
                    f'{axis}_location' in c or f'vertex_{axis}' in c
                )
            ][0]
            var_min, var_max = coords
            expr = ((pl.col(column)).is_in(np.arange(var_min, var_max+1,1)))
            table = table.filter(expr)
            # print(table)
    if isinstance(table, GDF):
        # print(table['geometry'])
        x_min, x_max = coords['x']
        y_min, y_max = coords['y']
        polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max),
            (x_min, y_min),
        ])
        check = table['geometry'].within(polygon)
        table = table[check]
        if table.index.name in table.columns:
            table.drop(table.index.name, axis=1, inplace=True)
        # print('sub_table:', table)
    return table


def relative(
    table,
    coords,
):
    assert depth(coords) == 1, \
        f'`coords` incorrect: {coords}\n Consult example .json!'
    if isinstance(table, PDF):
        for axis, coords in coords.items():
            column = [
                c for c in table.columns if (
                    f'{axis}_location' in c or f'vertex_{axis}' in c
                )
            ][0]
            var_min, _ = coords
            relatived = table.select(
                ((pl.col(column)) - var_min)
            )
            table = table.update(relatived)
    if isinstance(table, GDF):
        x_min, _ = coords['x']
        y_min, _ = coords['y']
        print('IN RELATIVE', table)
        geometry = table['geometry'].transform(
            lambda x: (x - np.array([x_min, y_min])).round(0)
        )
        table['geometry'] = geometry
    return table



def prepare_type(table):
    table_type = type(table)
    if table_type in [str, os.PathLike, PosixPath]:
        path = Path(table)
        # if path.suffix == '.gz' or path.suffix == 'zip':
        if path.suffix == '.parquet':
            table = pq.read_table(table)
        if path.suffix == '.csv':
            table = pd.read_csv(table)
        if path.suffix == '.geojson':
            table = gpd.read_file(table)
        table_type = type(table)

    if isinstance(table, TABLE):
        columns = list(table.column_names)
    else:
        columns = list(table.columns)

    if ('geometry' in columns):
        if isinstance(table, TABLE):
            table = gpd.GeoDataFrame.from_arrow(table)
        if isinstance(table, PDF):
            table = gpd.GeoDataFrame(table.to_pandas())
        if isinstance(table, DF):
            table = gpd.GeoDataFrame(table)
        assert isinstance(table, GDF), \
            f'Pipeline not equipped to convert {type(table)} to {GDF}'
    elif (('x_location' in columns or
        'vertex_x' in columns)):
        if isinstance(table, DF):
            table = pl.from_pandas(table)
        elif isinstance(table, TABLE):
            table = pl.from_arrow(table)
        assert isinstance(table, PDF), \
            f'Pipeline not equipped to convert {type(table)} to {PDF}'
    else:
        print(f'Something`s wrong.\n {type(table)}: {table}')
    return table


def wrap_table_actions(
    table: Union[str, os.PathLike, PosixPath, TABLE, GDF, PDF, DF],
    action: str,
    pixelsize_xy: float = None,
    coords: dict = None,
):
    table = prepare_type(table)
    print('Table Type: ', {type(table)})
    if pixelsize_xy:
        table = pixelate(
            table,
            pixelsize_xy
        )
    if action == 'location':
        table = filter_by_location(
            table=table,
            coords=coords
        )
    if action == 'relative':
        table = relative(
            table,
            coords
        )
    return table


def prepare_xenium_parquets(
    table: Union[TABLE, PDF],
    section: str,
    coords: dict,
    pixelsize_xy: float,
    output_path: Union[str, os.PathLike, PosixPath],
    bound: str = None
):
    schema = table.schema
    table = wrap_table_actions(
        table=table,
        coords=coords,
        pixelsize_xy=pixelsize_xy,
        action='location'
    )

    table = wrap_table_actions(
        table,
        'relative',
        coords=coords
    )

    output_path = Path(output_path / f'{section}')
    if bound:
        output_path = Path(output_path / f'boundaries/{bound}_relative.parquet')
    else:
        output_path = Path(output_path / 'transcripts/relative.parquet')

    table.write_parquet(
        file=output_path,
        compression='snappy',
        use_pyarrow=True,
        pyarrow_options=dict(
            version = '1.0',
            # schema=schema,
        ),
        # arrow_schema=schema
    )


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