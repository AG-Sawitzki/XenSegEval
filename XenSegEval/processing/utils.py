from XenSegEval.utils import (
    depth,
    get_section_coords
)

import os
from pathlib import Path

from shapely import (
    wkb,
    affinity,
    Polygon,
    # transform
)
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import geopandas as gpd

from typing import Any, Union
from pathlib import PosixPath


TABLE = pa.lib.Table
GDF = gpd.geodataframe.GeoDataFrame
PDF = pl.dataframe.frame.DataFrame
DF = pd.DataFrame


def pixelate(
    table: Union[PDF, GDF],
    pixelsize_xy: float,
)-> Union[PDF, GDF]:
    if isinstance(table, PDF):
        if 'x_location' in table.columns:
            target = 'location'
        if 'x_vertex' in table.columns:
            target = 'vertex' 
        pixeled = table.select(
            pl.col(f'x_{target}') / pixelsize_xy,
            pl.col(f'y_{target}') / pixelsize_xy,
        )
        table.update(pixeled)
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
    section_dict,
) -> Union[PDF, GDF]:
    assert depth(section_dict) == 1, \
        f'`section_dict` incorrect: {section_dict}\n Consult example .json!'
    if isinstance(table, PDF):
        for axis, coords in section_dict.items():
            column = [c for c in table.column_names if f'{axis}_' in c][0]
            var_min, var_max = coords
            expr = ((pl.field(column)).is_in(range(var_min, var_max+1,1)))
            sub_table = table.filter(expr)
    if isinstance(table, GDF):
        print(table['geometry'])
        x_min, x_max = section_dict['x']
        y_min, y_max = section_dict['y']
        polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max),
        ])
        check = table['geometry'].within(polygon)
        sub_table = table[check]
        if sub_table.index.name in sub_table.columns:
            sub_table.drop(sub_table.index.name, axis=1, inplace=True)
        print('sub_table:', sub_table)
    return sub_table


def relative(
    table,
    section_dict,
):
    assert depth(section_dict) == 1, \
        f'`section_dict` incorrect: {section_dict}\n Consult example .json!'
    if isinstance(table, PDF):
        for axis, coords in section_dict.items():
            column = [c for c in table.column_names if f'{axis}_' in c][0]
            var_min, _ = coords
            expr = ((pl.field(column)) - var_min)
            sub_table = table.filter(expr)
    if isinstance(table, GDF):
        x_min, _ = section_dict['x']
        y_min, _ = section_dict['y']
        print('IN RELATIVE', table)
        geometry = table['geometry'].transform(
            lambda x: (x - np.array([x_min, y_min])).round(0)
        )
        table['geometry'] = geometry
        sub_table = table
    return sub_table



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
    if (
        'geometry' in table.columns or
        'geometry' in table.column_names
    ):
        if isinstance(table, TABLE):
            table = gpd.GeoDataFrame.from_arrow(table)
        if isinstance(table, PDF):
            table = gpd.GeoDataFrame(table.to_pandas())
        if isinstance(table, DF):
            table = gpd.GeoDataFrame(table)
        assert isinstance(table, GDF), \
            f'Pipeline not equipped to convert {type(table)}'
    elif (
        ('x_location' in table.columns or
            'x_location' in table.column_names)
        or
        ('x_vertex' in table.columns or
            'x_vertex' in table.column_names)
    ):
        if isinstance(table, DF):
            table = pl.from_pandas(table)
        elif isinstance(table, TABLE):
            table = pl.from_arrow(table)
        assert isinstance(table, PDF), \
            f'Pipeline not equipped to convert {type(table)}'
    return table


def wrap_table_actions(
    table: Union[str, os.PathLike, PosixPath, TABLE, GDF, PDF, DF],
    action: str,
    pixelsize_xy: float = None,
    section_dict: dict = None,
):
    table = prepare_type(table)
    print('Table Type: ', {type(table)})
    if pixelsize_xy:
        table = pixelate(table, pixelsize_xy)
    if action == 'location':
        table = filter_by_location(
            table=table,
            section_dict=section_dict
        )
    if action == 'relative':
        table = relative(
            table,
            section_dict
        )
    return table


# def prepare_transcripts(
#     table,
#     section,
#     pixelsize_xy,
#     processed,
# ):
#     section, coords = section
#     table = wrap_table_actions(
#         table=table,
#         section=coords,
#         filter_type='location'
#     )

#     table.write_parquet(
        
#     )