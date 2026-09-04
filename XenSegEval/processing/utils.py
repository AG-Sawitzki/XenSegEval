from XenSegEval.utils import (
    depth,
    get_section_coords
)

import os
import gzip
import json
from math import sqrt
from pathlib import Path
from itertools import product
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
import matplotlib.colors as clrs

from typing import Any, Union
from pathlib import PosixPath
from numpy.typing import ArrayLike


TABLE = pa.lib.Table
GDF = gpd.geodataframe.GeoDataFrame
PDF = pl.dataframe.frame.DataFrame
DF = pd.DataFrame

MPG = geometry.multipolygon.MultiPolygon


def make_area_centres(
    n_roi: int,
    shape: tuple,
)-> list:
    """
    Fit `n_roi` squares equally distributed into `shape`.

    Relation between x & y decides squares per axis.
    Claculates the location of the centres of these squares.

    Parmeters
    ---------
        n_roi : int
            Number of expected punches/ROIs in an image.
        shape : tuple
            Shape of an Image with corresponding `n_roi`.

    Results
    -------
        out : list
            List of x,y coordinates. Centres of `n_roi` squares.

    """
    x,y = sorted(shape)
    print(x,y)
    partitions_x = int(sqrt(x/y*n_roi))
    partitions_y = int(n_roi/partitions_x)
    print(partitions_x, partitions_y)
    x_size = int(x/partitions_x)
    y_size = int(y/partitions_y)
    print(x_size, y_size)
    areas = list(product(
        range(0, y-y % y_size, y_size),
        range(0, x-x % x_size, x_size)
    ))
    centres = []
    for idx in range(len(areas)):
        coords_org = np.array(areas[idx])
        coords_ext = coords_org + np.array([x_size, y_size])
        print(coords_ext)
        centres.append(
            [coords_ext[1]-(coords_ext[1]-coords_org[1])//2,
             coords_ext[0]-(coords_ext[0]-coords_org[0])//2]
        )
    return centres


def get_area_index(
    centre: Union[list, tuple, ArrayLike],
    centres: Union[list, ArrayLike]
)-> int:
    """
    Retruns the Index of the nearest coordinates in a list.

    Parameters
    ----------
        centre : list, tuple or ArrayLike
            x,y coordinates.
        centres : list, ArrayLike
            List of x,y coordinates of centres of `n_roi` squares.

    Returns
    -------
        out : int
            Index of coordinates closest to `centre`.
            I.e. `n_roi` number.
    """
    relative_centres = np.abs(np.array(centres) - np.array(centre))
    tuple_relative_centres = [(coords[0], coords[1]) for coords in relative_centres]
    closest = min(tuple_relative_centres)
    idx = tuple_relative_centres.index(closest)
    return idx


def get_weighted_distance(
    centre: Union[tuple, list],
    weightx: float = 0.35,
    weighty: float = 1
) -> float:
    """
    Get weighted distance of an area's centre from [0,0].

    Parameters:
    ----------
        centre : tuple or list
            Centre of the area. Given in (y,x).
        weightx : float
            How large the impact of x is on the distance.
            lower x = similar y values have lower distance.
        weighty : float
            How large the impact of x,y is on the distance.
            lower y = similar x values have lower distance.

    Retruns
    ----------
        out : float
            Distance as float.
    """
    x, y = centre
    return np.sqrt((x*weightx)**2 + (y*weighty)**2)


def find_rois(
    shape_org: tuple,
    image_subres: ArrayLike,
    n_roi: int
) -> Union[list, ArrayLike]:
    """
    Sort the contours by area.

    Parameters:
    ----------
        shape_org : tuple
            Max resolution of img.
        image_subres : ArrayLike
            Lowest subresolution of image.
        n_roi : int
            Expected # of regions of interest.
            Should be equivalent to the number of tissue-samples on the slide.

    Returns
    ----------
        out : ArrayLike or list
            Contours of significant size.
    """
    z, y, x = shape_org

    subres_centre = np.uint8(image_subres[z//2])

    subres_dilated = cv2.dilate(
        subres_centre,
        np.ones((3, 3)),
        iterations=5
    )
    _, subres_binary = cv2.threshold(
        subres_dilated,
        127, 255, 0
    )

    contours, _ = cv2.findContours(
        subres_binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # keep contours with significant size
    values = []
    dtype = [('area', float), ('ai', float), ('y', float), ('x', float)]

    centres = make_area_centres(n_roi, subres_centre.shape)
    print(centres)
    for c in contours:
        (x, y), (w, h), a = cv2.minAreaRect(c)
        ai = get_area_index([x,y],centres)
        print(x,y, ai)
        values.append((w*h, ai, y, x))

    values_arr = np.array(values, dtype=dtype)
    # sort by area and find smallest allowed roi
    values_arr_sorted = np.sort(values_arr, kind='stable', order='area')
    smallest_allowed_roi = values_arr_sorted['area'][-n_roi]
    # sort by weighted_distance
    values_arr_ai_args = np.argsort(values_arr, kind='stable', order=['ai','y'])
    values_arr_ai_sorted = np.sort(values_arr, kind='stabe', order=['ai','y'])
    contours_ai_sorted = [contours[index] for index in values_arr_ai_args]

    mask = values_arr_ai_sorted['area'] >= smallest_allowed_roi
    nroi_contours = [
        contours_ai_sorted[index] for index, boolean in enumerate(mask)
        if boolean
    ]

    return nroi_contours, subres_centre


def check_colour(
    r: int,
    g: int,
    b: int
) -> tuple:
    """
    Gives a new rgb colour-tuple, incremented by 1.

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
    """
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
    """
    GeoJson Polygons to masks in a TIF.

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
    """
    r, g, b = (1, 0, 0)
    img = np.zeros(shape)
    img_aics = np.zeros_like(img, np.uint8)
    if type(layer) is int:
        gds = gdf[gdf['layer'] == layer]['geometry']
    else:
        gds = gdf['geometry']
    cells = 1
    for pg in gds:
        if isinstance(pg, MPG):
            for lr in pg.geoms:
                gmi = np.zeros_like(img, np.uint8)
                polyline = np.array(list(lr.exterior.coords))
                cv2.fillPoly(gmi, np.int32([polyline]), (r, g, b))
                cv2.fillPoly(img_aics, np.int32([polyline]), (r, g, b))
                r, g, b = check_colour(r, g, b)
                print(gmi)
                img[gmi > 0] = cells
                cells += 1
        else:
            gmi = np.zeros_like(img, np.uint8)
            polyline = np.array(list(pg.exterior.coords))
            cv2.fillPoly(gmi, np.int32([polyline]), (r, g, b))
            cv2.fillPoly(img_aics, np.int32([polyline]), (r, g, b))
            r, g, b = check_colour(r, g, b)
            img[gmi > 0] = cells
            cells += 1
    return img, img_aics


def wrap_ptm(
    polygons: Union[str, os.PathLike, PosixPath, GDF],
    output_path: Union[str, os.PathLike, PosixPath],
    shape: tuple,
    mode: str = None,
) -> None:
    """
    A wrapper for polygon_to_mask.

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
            Saves masks as `.tif` and `.npy` in output_dir.
    """
    if Path(polygons).suffix == '.gz':
        with gzip.open(polygons) as file:
            gdf = gpd.read_file(file)
    elif Path(polygons).suffix == '.geojson':
        gdf = gpd.read_file(polygons)
    elif isinstance(polygons, GDF):
        gdf = polygons
    else:
        print('input not path nor GeoDataFrame.')

    try:
        layers = max(gdf['layer'])
        for layer in range(layers+1):
            mask, mask_aics = polygon_to_mask(gdf, shape, layer)
            file = output_path / f'prediction_l{layer}.tif'
            file_aics = output_path / f'for_aics_l{layer}.tif'
            tifffile.imwrite(file, mask)
            tifffile.imwrite(file_aics, mask_aics)
            np.save(file.with_suffix('.npy'), mask)
            np.save(file_aics.with_suffix('.npy'), mask_aics)

    except KeyError:
        mask, mask_aics = polygon_to_mask(gdf, shape, layer=None)
        if mode:
            file = output_path / f'prediction_{mode}.tif'
            file_aics = output_path / f'for_aics_{mode}.tif'
        else:
            file = output_path / 'prediction.tif'
            file_aics = output_path / f'for_aics.tif'

        tifffile.imwrite(file, mask)
        tifffile.imwrite(file_aics, mask_aics)
        np.save(file.with_suffix('.npy'), mask)
        np.save(file_aics.with_suffix('.npy'), mask_aics)


def pixelate(
    table: Union[PDF, GDF],
    pixelsize_xy: float,
)-> Union[PDF, GDF]:
    """
    Transform coordinates in (geo)dataframe from length unit to pixelsize.

    Assumes pixel have same length in x & y. 

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to transform.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        pixelsize_xy : float
            Pixelsize of a pixel in xy plane.

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, but with adjusted coordinates.
    """
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
    coords: dict,
) -> Union[PDF, GDF]:
    """
    Keep entries in `table` that have coordinates in `coords`.

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to filter.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        coords : dict
            Dictionary containing min. x,y and max. x,y.
            For an example, see `gt_section.json`

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, excluding all entries with coordinates not in `coords`.
    """
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
    if isinstance(table, GDF):
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
    return table


def relative(
    table: Union[PDF, GDF],
    coords: dict,
)-> Union[PDF, GDF]:
    """
    Turn absolute coordinates into ones relative to the section origin.

    Parameters
    ----------
        table : polars or geopandas dataframe
            Contains the coordinates to adjust.
            Column with `vertex_` or `_location` for polars.
            `geometry` column in GeoDataFrame.
        coords : dict
            Dictionary containing min. x,y and max. x,y.
            For an example, see `gt_section.json`

    Returns
    -------
        out : polars or geopandas dataframe
            Same table as input, but with coordinates relative to section origin.
    """
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



def prepare_type(
    table: Union[TABLE, GDF, PDF, DF]
)-> Union[PDF, GDF]:
    """
    Changes the type of an input table dependent on the make-up.

    Parmeters
    ---------
        table : dataframe/table from pyarrow, geopandas, pandas or polars
            Input table to parse.
    Returns
    -------
        out : dataframe from geopandas or polars
            Output type depends on table-content.
    """
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
    action: str = None,
    pixelsize_xy: float = None,
    coords: dict = None,
)-> Union[GDF, PDF]:
    """
    Wrapper for functions above.

    Parameters
    ----------
        table : path to table or dataframe/table from pyarrow, geopandas, pandas or polars
            ...
        action : str
            Either: 
                'location' to filter by `coords`,
                'relative' to make coordinates in `table` relative to section origin,
                or none
        pixelsize_xy : float, optional
            Size of a pixel in x,y. If provided performes `pixelate`.
        coords : dict, optional
            Dictionary of section coordinates.
            See `gt_section.json` for an example.
            Required for `actions`: 'location' and 'relative'

    Returns
    -------
        out : polars or geopandas dataframe
            Input `table` converted and `action` performed. 
    """
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
)-> None:
    """
    Prepare Xenium parquets `{cell/nucleus}_boundaries.parquet` & `transcripts.parquet`.

    Parameters:
        table : pyarrow table or polars dataframe
            Parsed table to prepare.
        section : str
            Name of the section to prepare the table for
        coords : dict
            Dictionary of x,y coordinates of the section.
            See `gt_section.json` for an example.
        pixelsize_xy : float
            Size of a pixel in x,y direction.
        output_path : str, path
            Path to save the prepared table under.
        bound : str, optional
            Boundary type. Either 'cell' or 'nucleus'.
            Required if table is '_boundaries.parquet'.

    Retruns
    -------
        out : None
            Saves the table under `output_path`
    """
    # schema = table.schema
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
        output_path = Path(output_path / 'boundaries')
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path / f'{bound}_relative.parquet')
    else:
        output_path = Path(output_path / 'transcripts')
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path / 'relative.parquet')

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
    table.write_csv(
        file=output_path.with_suffix('.csv'),
    )


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


# function form stackoverflow
# adapted to return shapely Polygons
def process_roi(npy_data, npy_base_output_path):
    """
    Get the polgyons from the prediction-masks using cellpose.utils functions.
    
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
    """
    print(' - Extracting ROI')
    try:
        masks = npy_data.item().get("masks")
    except:
        masks = npy_data
    masks = masks.squeeze()
    # change the index order:
    # first the cell then the layer it is on.
    # thus one would now how the same cell looks on different layers
    data = {'layer': [], 'name': [], 'geometry': []}
    if masks.ndim == 3:
        zs = min(masks.shape)
        posZ = masks.shape.index(zs)
        for z in range(zs):
            if posZ == 0:
                mask_z = masks[z,...]
            elif posZ == 1:
                mask_z = masks[:,z,:]
            else:
                mask_z = masks[...,z]
            print(f' - Layer {z}')
            coords_list = outlines_list(mask_z)
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