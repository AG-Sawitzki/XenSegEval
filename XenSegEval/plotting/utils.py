import os
import gzip
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

from typing import Union
from pathlib import PosixPath
from geopandas.geodataframe import GeoDataFrame
from numpy.typing import ArrayLike


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
    eval_path = f'{results}/{method}/evaluation/{section}/'
    u4n_path = f'{eval_path}/results.csv'
    cs_path = f'{eval_path}/CS-BENCH.csv'
    df = pd.read_csv(u4n_path)
    if 'Method' in df.columns:
        data = np.array(df[['F1', 'Jaccard']])
        tick_labels = list(np.round(df['Threshold'], 2))
    df = pd.read_csv(cs_path)
    if len(df) == 1:
        data = np.vstack((data, np.array(df[['f1', 'jaccard']])))
        tick_labels.append('cs')
    ax.grouped_bar(data, tick_labels=tick_labels, labels=['F1', 'Jaccard'])
    ax.legend()
    ax.set_title(method)

    return None


def get_data(u4n, cs, u4n_path, cs_path, u4n_val, cs_vals):
    df_u4n = pd.read_csv(u4n_path)
    df_cs = pd.read_csv(cs_path)

    if u4n.shape == (0,):
        u4n = np.array(df_u4n[[u4n_val]]).T
        cs = np.array(df_cs[cs_vals])
    else:
        data_u4n = np.array(df_u4n[[u4n_val]]).T
        data_cs = np.array(df_cs[cs_vals])
        u4n = np.vstack((u4n, data_u4n))
        cs = np.vstack((cs, data_cs))

    return u4n, cs


def bar_compare_eval(
    methods,
    results,
    section,
    fig_u4n,
    fig_cs,
    ax_u4n,
    ax_cs,
):
    u4n_val = 'F1'
    cs_vals = ['f1', 'seg', 'jaccard', 'dice', 'PQ']

    u4n = np.array([])
    cs = np.array([])

    tick_labels = []
    for method in methods:
        eval_path = f'{results}/{method}/evaluation/{section}/'
        subdirs = list(Path(eval_path).glob('_*'))
        if len(subdirs) != 0:
            for subdir in subdirs:
                u4n_path = Path(f'{subdir}/results.csv')
                cs_path = Path(f'{subdir}/CS-BENCH.csv')
                if u4n_path.is_file() and cs_path.is_file():
                    tick_labels.append(method+subdir.stem)
                    u4n, cs = get_data(
                        u4n, cs, u4n_path, cs_path, u4n_val, cs_vals
                    )
        else:
            tick_labels.append(method)
            u4n_path = f'{eval_path}/results.csv'
            cs_path = f'{eval_path}/CS-BENCH.csv'
            u4n, cs = get_data(
                u4n, cs, u4n_path, cs_path, u4n_val, cs_vals
            )
    labels = np.round(np.arange(0.5, 0.95, 0.05), 2)
    ax_u4n.grouped_bar(u4n, tick_labels=tick_labels, labels=labels)
    ax_cs.grouped_bar(cs, tick_labels=tick_labels, labels=cs_vals)

    ax_u4n.tick_params(axis='x', rotation=35)
    ax_cs.tick_params(axis='x', rotation=35)

    ax_u4n.legend()
    ax_cs.legend()

    fig_u4n.savefig('/data/cephfs-1/home/users/juno12_c/test_u4n.png')
    fig_cs.savefig('/data/cephfs-1/home/users/juno12_c/test_cs.png')


def polygon_overlay(
    polygons: Union[str, os.PathLike, PathPosix, GeoDataFrame],
    img: Union[str, os.PathLike, PathPosix, ArrayLike],
    output_path: Union[str, os.PathLike, PathPosix],
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

    plt.style.use('./segment_style.mplstyle ')

    fz = 48
    dimy, dimx, c = img.shape

    fig.set_frameon(False)
    fig.set_size_inches(dimy/100, dimx/100)
    ax.tick_params(axis='both', which='major', labelsize=fz)
    ax.set_xlabel('x_location in px', fontsize=fz)
    ax.set_ylabel('y_location in px', fontsize=fz)

    ax.set_aspect('equal', 'box')

    asb = AnchoredSizeBar(
        ax.transData,
        size=941.1764705882354,
        label='200 µm',
        loc='lower left',
        frameon=False,
        size_vertical=47.05882352941177,
        color='white',
        fontproperties=FontProperties(size=fz)
    )
    ax.add_artist(asb)

    ax.set_xlim(0, dimx)
    ax.set_ylim(0, dimy)

    ax.imshow(img_norm)

    gdf.boundary.plot(
        ax=ax, aspect='equal', color='white'
    )

    fig.savefig(
        Path(output_path) / 'outline.png',
        dpi=100, bbox_inches='tight', pad_inches=0.0
    )
