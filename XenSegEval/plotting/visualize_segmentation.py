import matplotlib.pyplot as plt
from XenSegEval.plot import polygon_overlay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Ovrl.')
    parser.add_argument('-c', '--Config', help='Path to the config file.')
    parser.add_argument('-m', '--Method', help='Method to evaluate.')
    args = parser.parse_args()

    method = args.Method
    config_path = args.Config

    with open(config_path, 'rb') as f:
        config = tomlkit.load(f)

    variables = get_config_args(config, 'eval')
    globals().update(variables)

    polygons = '/data/cephfs-2/unmirrored/groups/sawitzki/Juno/TMA2/results/proseg/output/newmem/cell-polygons.geojson.gz'

    img = '/data/cephfs-2/unmirrored/groups/sawitzki/Juno/TMA2/processed/newmem/morphology/focus/focus.ome.tif'

    output_path = '/data/cephfs-1/home/users/juno12_c'

    fig, ax = plt.subplots()

    polygon_overlay(polygons, img, output_path, fig, ax)
