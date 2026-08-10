from shapely import wkb
import polars as pl
import geopandas as gpd

from pyarrow.lib import Table
from typing import Any, Union


def filter_by_location(
    table: Table,
    axis: str,
    min_max: Union[list, tuple]
) -> Table:
    column = [c for c in table.column_names if f'{axis}_' in c][0]
    var_min, var_max = min_max
    expr = ((pl.field(column)).is_in(range(var_min, var_max+1,1)))
    sub_table = table.filter(expr)
    return sub_table


def parquet_to_geodataframe(
    table: Table,
):
    df = table.to_pandas()
    df['geometry'] = gpd.GeoSeries.from_wkb(df['geometry'])
    gdf = gpd.GeoDataframe(df, geometry='geometry')
    return gdf


def wrap_parquet_actions(
    table,
    section,
    filter_type,
):
    if filter_type == 'location':
        for axis, coordinates in section.items():
            table = filter_by_location(
                table=table,
                axis=axis,
                min_max=coordinates
            )
        return table
    if filter_type == 'wkb':
        if isinstance(table, Table):
            gdf = parquet_to_geodataframe(table)
            return gdf


def prepare_transcripts(
    table,
    section,
    pixelsize_xy
    processed,
):
    section, coords = section
    table = wrap_parquet_actions(
        table=table,
        section=coords,
        filter_type='location'
    )

    table.write_parquet(
        
    )