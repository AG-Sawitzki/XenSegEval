from XenSegEval.utils import (
    get_config_args
)
from XenSegEval.processing.utils import (
    wrap_table_actions
)

import argparse
import tomlkit
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Polygon


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='boundaries')
    parser.add_argument(
        '-c', '--Config',
        default='config.toml',
        help='Path to the config file.'
    )

    args = parser.parse_args()

    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'boundaries')
    globals().update(variables)


    for section in sections:
        boundaries = f'{processed}/{section}/boundaries/'
        for file in Path(boundaries).glob('*_relative.parquet'):
            bound = str(file).removesuffix('_relative.parquet')
            bound = bound[bound.rfind('/')+1:]
            print(bound)
            cell_ids = []
            geometries = []

            parquet_file = pq.ParquetFile(file)

            df = parquet_file.read().to_pandas()

            for cell_id in set(df['cell_id']):
                x = df[df['cell_id'] == cell_id]['vertex_x']
                y = df[df['cell_id'] == cell_id]['vertex_y']
                if len(x) >= 4 or len(y) >= 4:
                    geometry = Polygon(zip(x,y))
                    cell_ids.append(cell_id)
                    geometries.append(geometry)
                # else:
                #     print(cell_id)
                #     print('x: ', x)
                #     print('y: ', y)

            data = dict(
                cell_id=cell_ids,
                geometry=geometries
            )

            gdf = gpd.GeoDataFrame(data)

            # sub_gdf = wrap_table_actions(
            #     gdf,
            #     'location',
            #     pixelsize_xy = pixelsizeXY,
            #     section_dict = section_dictionary[section]
            # )

            # print(sub_gdf)

            gj = gdf.to_json(na='keep', drop_id=True)

            output_path = Path(
                f'{results}/xenium/output/{section}/'
            )
            output_path.mkdir(parents=True, exist_ok=True)
            print(output_path)
            with open(Path(output_path / f'{bound}_polygons.geojson'), 'w') as f:
                f.write(gj)

