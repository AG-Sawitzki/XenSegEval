import pyarrow as pa
import pyarrow.parquet as pq
import geopandas
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

    boundaries = f'{processed}/{section}/boundaries/'

    for file in Path(boundaries).glob('*_boundaries.parquet'):
        bound = str(file).removesuffix('_boundaries.parquet')
        bound = str(bound).removeprefix(str(boundaries)+'/')
        cell_ids = []
        geometries = []

        file_path = f'{processed}/{section}/boundaries/{bound}_relative.parquet'

        parquet_file = pq.ParquetFile(file)

        df = parquet_file.read().to_pandas()

        for cell_id in set(df['cell_id']):
            x = df[df['cell_id'] == cell_id]['vertex_x']
            y = df[df['cell_id'] == cell_id]['vertex_y']
            if len(x) >= 4 or len(y) >= 4:
                geometry = Polygon(zip(x,y))
                cell_ids.append(cell_id)
                geometries.append(geometry)
            else:
                print(cell_id)
                print('x: ', x)
                print('y: ', y)

        data = dict(
            cell_id=cell_ids,
            geometry=geometries
        )

        gdf = geopandas.GeoDataFrame(data)

        gj = gdf.to_json(na='keep', drop_id=True)

        with open(file.replace('parquet','geojson'), 'w') as f:
            f.write(gj)

