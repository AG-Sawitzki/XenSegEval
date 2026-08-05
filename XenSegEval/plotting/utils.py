import os
import gzip
import json
from pathlib import Path

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

import tifffile
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Any, Union
from pathlib import PosixPath
from geopandas.geodataframe import GeoDataFrame
from numpy.typing import ArrayLike


colors = dict(
    cpsam='#FFB000',
    dinocell='#FE6100',
    dissect='#DC267F',
    mesmer='#785EF0',
    proseg='#648FFF',
    stardist='#79AB59',
)
'''Colors for the Algorithms.'''


def hex_to_rgb(color, reduce):
    color = color.lstrip('#')
    r = int(color[0:2], 16)*(1-reduce)
    g = int(color[2:4], 16)*(1-reduce)
    b = int(color[4:6], 16)*(1-reduce)

    return tuple(r, g, b)


def assign_color(colors, method, label, color, reduce):
    new_color = hex_to_rgb(color, reduce)
    pos = list(colors.keys().index(method))
    items = list(colors.items())
    items.insert(pos, (label, new_color))
    return dict(items)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, rotation_mode="xtick")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def bar_method_eval(
    fig,
    ax,
    results,
    method,
    section
):
    '''Plots the evaluation results from cs and/or u4n in a bar plot for a
        sepicified method.

    Parameters
    ----------
        fig : figure
            A figure to plot onto.
        ax : ax
            The Axes of the figure.
        results : Path 
            Path to the results folder.
        method : str
            Name of the method. Same as the name in config.toml
        section : str or int
            ROI segmentation was performed on.

    Returns
    ----------
        out : None
            Plots the bars.
    '''
    eval_path = f'{results}/{method}/evaluation/{section}/'
    u4n_path = f'{eval_path}/results.csv'
    cs_path = f'{eval_path}/CS-BENCH.csv'
    if Path(u4n_path).is_file():
        df = pd.read_csv(u4n_path)
        if 'Method' in df.columns:
            data = np.array(df[['F1', 'Jaccard']])
            tick_labels = list(np.round(df['Threshold'], 2))
    else:
        data = np.array([np.nan, np.nan])

    if Path(cs_path).is_file():
        df = pd.read_csv(cs_path)
        if len(df) == 1:
            data = np.vstack((data, np.array(df[['f1', 'jaccard']])))
            tick_labels.append('cs')
    else:
        data = np.vstack((data, np.array([np.nan, np.nan])))

    ax.grouped_bar(data, tick_labels=tick_labels, labels=['F1', 'Jaccard'])
    ax.legend()
    ax.set_title(method)

    return None


def get_data(
    arr,
    path,
    vals,
):
    '''Get the metrics from a DataFrame.

    Parameters
    ----------
        arr : ArrayLike
            The array to contain the metrics for plotting.
        path : Path
            Path to the DataFrame of the metrics.
        vals : str or list
            Values, i.e. Metrics, of interest.
    '''
    df = pd.read_csv(path)

    if arr.shape == (0,):
        if len(vals) > 1:
            arr = np.array(df[[vals]]).T
        else:
            arr = np.array(df[vals])
    else:
        if len(vals) > 1:
            data = np.array(df[[vals]]).T
        else:
            data = np.array(df[vals])
        arr = np.vstack((arr, data))

    return arr


def bar_compare_eval(
    methods: list,
    results: Union[str, os.PathLike, PosixPath],
    section: str,
    fig: Any,
    ax: Any,
    colors: Union[dict, str, os.PathLike, PosixPath],
    eval: str = 'cs',
):
    if type(colors) is not dict:
        with open(colors, 'r') as f:
            colors = json.load(f)

    if method == 'u4n':
        vals = 'F1'
        tick_labels = np.round(np.arange(0.5, 0.95, 0.05), 2)
        file = 'results.csv'
    else:
        vals = ['f1', 'seg', 'jaccard', 'dice', 'PQ']
        tick_labels = vals
        file = 'CS-BENCH.csv'

    arr = np.array([])

    labels = []
    for method in methods:
        eval_path = f'{results}/{method}/evaluation/{section}/'
        subdirs = list(Path(eval_path).glob('_*'))
        if len(subdirs) != 0:
            i = 1
            for subdir in subdirs:
                path = Path(f'{subdir}/{file}')
                if path.is_file():
                    label = method+'_'+subdir.stem
                    labels.append(label)
                    colors = assign_color(
                        colors, method, label,
                        color=colors[method],
                        reduce=0.2*i,
                    )
                    arr = get_data(
                        arr, arr, vals
                    )
                i += 1
            del colors[method]
        else:
            labels.append(method)
            path = f'{eval_path}/{file}'
            arr = get_data(
                arr, arr, vals
            )
    color_list = [value for key, value in colors.items()]
    ax.grouped_bar(arr, tick_labels=tick_labels, labels=labels, colors=color_list)
    ax.tick_params(axis='x', rotation=35)
    ax.legend()
    ax.savefig(f'/data/cephfs-1/home/users/juno12_c/test_{eval}.pdf')


def polygon_overlay(
    polygons: Union[str, os.PathLike, PosixPath, GeoDataFrame],
    img: Union[str, os.PathLike, PosixPath, ArrayLike],
    output_path: Union[str, os.PathLike, PosixPath],
    fig,
    ax,
    pixelsize_xy=0.2125,
    **kwargs
) -> None:
    if ax is None:
        ax = plt.gca()

    if type(polygons) is not GeoDataFrame:
        if Path(polygons).suffix == '.gz':
            with gzip.open(polygons) as file:
                gdf = gpd.read_file(file)
        else:
            gdf = gpd.read_file(polygons)
    else:
        gdf = polygons

    if type(img) is not np.ndarray:
        img = tifffile.imread(img)

    assert type(img) is np.ndarray, f'Img has type {type(img)}.'
    assert type(gdf) is GeoDataFrame, f'Polygons has {type(gdf)}.'

    if img.shape[-1] == 4:
        img = img[..., :3]

    img_norm = (img-img.min())/(img.max()-img.min())

    # plt.style.use('./segment_style.mplstyle ')

    fz = 48
    dimy, dimx, c = img.shape

    fig.set_frameon(False)
    fig.set_size_inches(dimy/100, dimx/100)
    ax.tick_params(axis='both', which='major', labelsize=fz)
    ax.set_xlabel('x_location in px', fontsize=fz)
    ax.set_ylabel('y_location in px', fontsize=fz)

    ax.set_aspect('equal', 'box')

    size = dimx/10
    length = np.round(size*pixelsize_xy, 0)

    asb = AnchoredSizeBar(
        ax.transData,
        size=size,
        label=f'{length} µm',
        loc='lower left',
        frameon=False,
        size_vertical=10/pixelsize_xy,
        color='white',
        fontproperties=FontProperties(size=fz)
    )
    ax.add_artist(asb)

    ax.set_xlim(0, dimx)
    ax.set_ylim(dimy, 0)

    ax.imshow(img_norm)

    gdf.boundary.plot(
        ax=ax, aspect='equal', color='white'
    )

    fig.savefig(
        Path(output_path),
        dpi=250, bbox_inches='tight', pad_inches=0.0
    )



# add a function to asign colors based on config?
# below from IBM
# \definecolor{cpsam}{HTML}{FFB000}
# \definecolor{dinocell}{HTML}{FE6100}
# \definecolor{dissect}{HTML}{DC267F}
# \definecolor{mesmer}{HTML}{785EF0}
# \definecolor{proseg}{HTML}{648FFF}
# \definecolor{stardist}{HTML}{79AB59}
# below from BIH
# \definecolor{blau}{HTML}{003754}
# \definecolor{weiss}{HTML}{FFFFFF}
# \definecolor{schwarz}{HTML}{000000}
# \definecolor{hellrosa}{HTML}{FFB0AC}
# \definecolor{dunkelrot}{HTML}{AF1821}
# \definecolor{korall}{HTML}{EA5451}
# \definecolor{gold}{HTML}{9D7220}
# \definecolor{mineral}{HTML}{009AA9}
# \definecolor{lavendel}{HTML}{7876B6}
