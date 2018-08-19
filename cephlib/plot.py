import logging
import warnings
from io import BytesIO
from functools import wraps
from typing import Tuple, cast, List, Callable, Optional, Any

import numpy
import scipy.stats
import matplotlib.style
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import gridspec, ticker
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from .common import float2str
from .numeric import auto_edges
from .types import array1d
from .numeric_types import ndarray2d, ndarray1d
from .statistic import (moving_average, moving_dev, hist_outliers_perc, find_ouliers_ts, approximate_curve,
                        StatProps, NormStatProps)
from .numeric_types import DataSource, TimeSeries
from .units import unit_conversion_coef_f
from .istorage import IImagesStorage


logger = logging.getLogger('cephlib')


def hmap_from_2d(data: numpy.ndarray, max_xbins: int = 25,
                 noval: Any = None) -> Tuple[ndarray1d, List[Tuple[int, int]]]:
    """
    :param data: 2D array of input data, [num_measurement * num_objects], each row contains single measurement for
                 every object
    :param max_xbins: maximum time ranges to split to
    :param noval: Optional, if not None - faked value, which mean "no measurement done, or measurement invalid",
                  removed from results array
    :return: pair if 1D array of all valid values and list of pairs of indexes in this
             array, which belong to one time slot
    """
    assert len(data.shape) == 2

    # calculate how many 'data' columns fit into single output interval
    num_points = data.shape[1]
    step = int(round(float(num_points) / max_xbins + 0.5))

    # drop last columns, as in other case it's hard to make heatmap looks correctly
    idxs = range(0, num_points, step)

    # list of chunks of data array, which belong to same result slot, with noval removed
    result_chunks = []
    for bg, en in zip(idxs[:-1], idxs[1:]):
        block_data = data[:, bg:en]
        filtered = block_data.reshape(block_data.size)
        if noval is not None:
            filtered = filtered[filtered != noval]
        result_chunks.append(filtered)

    # generate begin:end indexes of chunks in chunks concatenated array
    chunk_lens = numpy.cumsum(list(map(len, result_chunks)))
    bin_ranges = list(zip(chunk_lens[:-1], chunk_lens[1:]))

    return numpy.concatenate(result_chunks), bin_ranges


def process_heatmap_data(values: numpy.ndarray,
                         bin_ranges: List[Tuple[float, float]],
                         cut_percentile: Tuple[float, float] = (0.02, 0.98),
                         ybins: int = 20,
                         log_edges: bool = True,
                         bins: ndarray1d = None) -> Tuple[ndarray2d, ndarray1d]:
    """
    Transform 1D input array of values into 2D array of histograms.
    All data from 'values' which belong to same region, provided by 'bin_ranges' array
    goes to one histogram

    :param values: 1D array of values - this array gets modified if cut_percentile provided
    :param bin_ranges: List of pairs begin:end indexes in 'values' array for items, belong to one section
    :param cut_percentile: Options, pair of two floats - clip items from 'values', which not fit into provided
                           percentiles (100% == 1.0)
    :param ybins: ybin count for histogram
    :param log_edges: use logarithmic scale for histogram bins edges
    :return: 2D array of heatmap
    """
    assert len(values.shape) == 1

    nvalues = [values[idx1:idx2] for idx1, idx2 in bin_ranges]

    if cut_percentile:
        mmin, mmax = numpy.percentile(values, (cut_percentile[0] * 100, cut_percentile[1] * 100))
        numpy.clip(values, mmin, mmax, values)
    else:
        mmin = values.min()
        mmax = values.max()

    if bins is None:
        if log_edges:
            bins = auto_edges(values, bins=ybins, round_base=None)
        else:
            bins = numpy.linspace(mmin, mmax, ybins + 1)
    else:
        nvalues = numpy.clip(nvalues, bins[0], bins[-1])

    return numpy.array([numpy.histogram(src_line, bins)[0] for src_line in nvalues]), bins


def plot_histo(ax: Axes, vals: array1d, bins: array1d = None, kde: bool = False,
               left: float = None, right: float = None,
               xlabel: str = None, y_ticks: bool = False):
    assert len(vals.shape) == 1
    seaborn.distplot(vals, bins=bins, ax=ax, kde=kde, axlabel=xlabel)

    if not y_ticks:
        ax.set_yticklabels([])

    if left is not None or right is not None:
        ax.set_xlim(left=left, right=right)


def plot_hmap_with_histo(fig: Figure, data: ndarray1d, chunk_ranges: List[Tuple[float, float]],
                         bins: ndarray1d = None, **args):
    assert len(data.shape) == 1
    heatmap, bins = process_heatmap_data(data, chunk_ranges, bins=bins)
    bins_populations, _ = numpy.histogram(data, bins)
    return do_plot_hmap_with_histo(fig, heatmap, bins_populations, bins, **args)


def do_plot_hmap_with_histo(fig: Figure,
                            heatmap: ndarray2d,
                            bins_populations: ndarray1d,
                            bins: ndarray1d,
                            cmap: str = None,
                            cbar: bool = False,
                            avg_labels: bool = False,
                            histo_grid: Optional[str] = 'y'):
    assert len(heatmap.shape) == 2

    gs = gridspec.GridSpec(1, 10)
    ax = fig.add_subplot(gs[0, :6])
    seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap=cmap, ax=ax, cbar=cbar)
    ax.axhline(linewidth=1, color="b")

    if avg_labels:
        lbins = list((bins[:-1] + bins[1:]) / 2) + [bins[-1]]
        labels = list(map(float2str, lbins))
        labels[-1] += '+'
        label_locs = numpy.arange(len(labels)) + 0.5
    else:
        labels = list(map(float2str, bins))
        label_locs = numpy.arange(len(labels))

    ax.yaxis.tick_right()
    ax.yaxis.set_major_locator(ticker.FixedLocator(label_locs))
    ax.set_yticklabels(labels[::-1], rotation='horizontal')

    ax2 = fig.add_subplot(gs[0, 7:])
    ax2.set_yticklabels([])
    ax2.set_ylim(top=len(bins_populations), bottom=0)
    bins_populations_perc = bins_populations * 100 / bins_populations.sum()
    ax2.barh(numpy.arange(len(bins_populations_perc)) + 0.5, width=bins_populations_perc)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: float2str(v, digits=2)))

    if isinstance(histo_grid, str):
        ax2.grid(axis=histo_grid)
    else:
        ax2.grid(histo_grid)

    return ax, ax2

# --------------  PLOT HELPERS FUNCTIONS  ------------------------------------------------------------------------------

def get_emb_image(fig: Figure, file_format: str, **opts) -> bytes:
    bio = BytesIO()
    if file_format == 'svg':
        fig.savefig(bio, format='svg', **opts)
        img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().decode("utf8").split(img_start, 1)[1].encode("utf8")
    else:
        fig.savefig(bio, format=file_format, **opts)
        return bio.getvalue()


class PlotParams:
    def __init__(self, fig: Figure, ax: Any, title: str, style: Any, colors: Any) -> None:
        self.fig = fig
        self.ax = ax
        self.style = style
        self.colors = colors
        self.title = title


def provide_plot(noaxis: bool = False,
                 eng: bool = False,
                 no_legend: bool = False,
                 long_plot: bool = True,
                 grid: Any = None,
                 style_name: str = 'default',
                 noadjust: bool = False) -> Callable[..., Callable[..., str]]:
    def closure1(func: Callable[..., None]) -> Callable[..., str]:
        @wraps(func)
        def closure2(storage: IImagesStorage,
                     style: Any,
                     colors: Any,
                     path: DataSource,
                     title: Optional[str],
                     *args, **kwargs) -> str:
            fpath = storage.check_plot_file(path)
            if not fpath:

                assert style_name in ('default', 'ioqd')
                mlstyle = style.default_style if style_name == 'default' else style.io_chart_style
                with matplotlib.style.context(mlstyle):
                    file_format = path.tag.split(".")[-1]
                    fig = plt.figure(figsize=style.figsize_long if long_plot else style.figsize)
                    if not noaxis:
                        xlabel = kwargs.pop('xlabel', None)
                        ylabel = kwargs.pop('ylabel', None)
                        ax = fig.add_subplot(111)

                        if xlabel is not None:
                            ax.set_xlabel(xlabel)

                        if ylabel is not None:
                            ax.set_ylabel(ylabel)

                        if grid:
                            if grid is True:
                                ax.grid(True)
                            else:
                                ax.grid(axis=grid)
                    else:
                        ax = None

                    if title:
                        fig.suptitle(title, fontsize=style.title_font_size)

                    pp = PlotParams(fig, ax, title, style, colors)
                    func(pp, *args, **kwargs)
                    apply_style(pp, eng=eng, no_legend=no_legend, noadjust=noadjust)

                    img = get_emb_image(fig, file_format=file_format, dpi=style.dpi)
                    fpath = storage.put_plot_file(img, path)
                    logger.debug("Plot %s saved to %r", path, fpath)
                    plt.close(fig)
            return fpath
        return closure2
    return closure1


def apply_style(pp: PlotParams, eng: bool = True, no_legend: bool = False, noadjust: bool = False) -> None:

    if (pp.style.legend_for_eng or not eng) and not no_legend:
        if not noadjust:
            pp.fig.subplots_adjust(right=pp.style.subplot_adjust_r)
        legend_location = "center left"
        legend_bbox_to_anchor = (1.03, 0.81)

        for ax in pp.fig.axes:
            ax.legend(loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    elif not noadjust:
        pp.fig.subplots_adjust(right=pp.style.subplot_adjust_r_no_legend)

    if pp.style.tide_layout:
        pp.fig.set_tight_layout(True)


# --------------  PLOT FUNCTIONS  --------------------------------------------------------------------------------------


@provide_plot(eng=True)
def plot_hist(pp: PlotParams, units: str, prop: StatProps) -> None:

    normed_bins = prop.bins_populations / prop.bins_populations.sum()
    bar_width = prop.bins_edges[1] - prop.bins_edges[0]
    pp.ax.bar(prop.bins_edges, normed_bins, color=pp.colors.box_color, width=bar_width, label="Real data")

    pp.ax.set(xlabel=units, ylabel="Value probability")

    if isinstance(prop, NormStatProps):
        nprop = cast(NormStatProps, prop)
        stats = scipy.stats.norm(nprop.average, nprop.deviation)

        new_edges, step = numpy.linspace(prop.bins_edges[0], prop.bins_edges[-1],
                                         len(prop.bins_edges) * 10, retstep=True)

        ypoints = stats.cdf(new_edges) * 11
        ypoints = [nextpt - prevpt for (nextpt, prevpt) in zip(ypoints[1:], ypoints[:-1])]
        xpoints = (new_edges[1:] + new_edges[:-1]) / 2

        pp.ax.plot(xpoints, ypoints, color=pp.colors.primary_color, label="Expected from\nnormal\ndistribution")

    pp.ax.set_xlim(left=prop.bins_edges[0])
    if prop.log_bins:
        pp.ax.set_xscale('log')


@provide_plot(grid='y')
def plot_simple_over_time(pp: PlotParams, tss: List[Tuple[str, numpy.ndarray]], average: bool = False) -> None:
    max_len = 0
    for name, arr in tss:
        if average:
            avg_vals = moving_average(arr, pp.style.avg_range)
            if pp.style.approx_average_no_points:
                time_points = numpy.arange(len(avg_vals))
                avg_vals = approximate_curve(cast(List[int], time_points),
                                             avg_vals,
                                             cast(List[int], time_points),
                                             pp.style.curve_approx_level)
            arr = avg_vals
        pp.ax.plot(arr, label=name)
        max_len = max(max_len, len(arr))
    pp.ax.set_xlim(-5, max_len + 5)


@provide_plot(no_legend=True, grid='x', noadjust=True)
def plot_simple_bars(pp: PlotParams,
                     names: List[str],
                     values: List[float],
                     errs: List[float] = None,
                     x_formatter: Callable[[float, float], str] = None,
                     one_point_zero_line: bool = True) -> None:

    ind = numpy.arange(len(names))
    width = 0.35
    pp.ax.barh(ind, values, width, xerr=errs)

    pp.ax.set_yticks(ind)
    pp.ax.set_yticklabels(names)
    pp.ax.set_xlim(0, max(val + err for val, err in zip(values, errs)) * 1.1)

    if one_point_zero_line:
        pp.ax.axvline(x=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)

    if x_formatter:
        pp.ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

    pp.fig.subplots_adjust(left=0.2)


@provide_plot(no_legend=True, grid=True)
def plot_dots_with_regression(pp: PlotParams, x: array1d, y: array1d,
                              x_approx: array1d = None, y_approx: array1d = None) -> None:
    pp.ax.plot(x, y, '.')
    if x_approx is not None:
        pp.ax.plot(x_approx, y_approx, '--')


@provide_plot(no_legend=True, long_plot=True, noaxis=True)
def plot_hmap_from_2d(pp: PlotParams, data2d: numpy.ndarray, xlabel: str, ylabel: str,
                      bins: numpy.ndarray = None) -> None:
    ioq1d, ranges = hmap_from_2d(data2d)
    heatmap, bins = process_heatmap_data(ioq1d, bin_ranges=ranges, bins=bins)
    bins_populations, _ = numpy.histogram(ioq1d, bins)

    ax, _ = do_plot_hmap_with_histo(pp.fig,
                                    heatmap,
                                    bins_populations,
                                    bins,
                                    cmap=pp.colors.hmap_cmap,
                                    cbar=pp.style.heatmap_colorbar,
                                    histo_grid=pp.style.histo_grid)
    ax.set(ylabel=ylabel, xlabel=xlabel)


@provide_plot(eng=True, grid='y')
def plot_v_over_time(pp: PlotParams, units: str, ts: TimeSeries,
                     plot_avg_dev: bool = True, plot_points: bool = True) -> None:
    # convert time to ms
    if len(ts.times) != len(ts.data):
        import IPython
        IPython.embed()
    assert ts.times.min() == ts.times[0]
    assert len(ts.times) == len(ts.data)

    time_points = (ts.times - ts.times[0]) * unit_conversion_coef_f(ts.time_units, 's')
    outliers_idxs = find_ouliers_ts(ts.data, cut_range=pp.style.outliers_q_nd)
    outliers_4q_idxs = find_ouliers_ts(ts.data, cut_range=pp.style.outliers_hide_q_nd)
    normal_idxs = numpy.logical_not(outliers_idxs)

    outl_4q_count = numpy.count_nonzero(outliers_4q_idxs)
    if outl_4q_count != 0 and outl_4q_count < pp.style.max_hidden_outliers_fraction * len(ts.data):
        outliers_idxs = outliers_idxs & numpy.logical_not(outliers_4q_idxs)
    else:
        outliers_4q_idxs = None

    data = ts.data[normal_idxs]
    data_times = time_points[normal_idxs]

    if plot_points:
        outliers = ts.data[outliers_idxs]
        outliers_times = time_points[outliers_idxs]

        alpha = pp.colors.noise_alpha if plot_avg_dev else 1.0
        pp.ax.plot(data_times, data, pp.style.point_shape, color=pp.colors.primary_color, alpha=alpha, label="Data")

        if len(outliers) > 0:
            if outliers_4q_idxs is not None:
                label = "{}Q < Outliers < {}Q".format(pp.style.outliers_q_nd, pp.style.outliers_hide_q_nd)
            else:
                label = "{}Q+ Outliers".format(pp.style.outliers_q_nd)
            pp.ax.plot(outliers_times, outliers, pp.style.err_point_shape, color=pp.colors.err_color, label=label)

        if outliers_4q_idxs is not None:
            hidden_outliers = ts.data[outliers_4q_idxs]
            hidden_outliers_times = time_points[outliers_4q_idxs]

            med = numpy.median(data)

            if len(outliers) > 0:
                max_val = max(data.max(), outliers.max())
                min_val = min(data.min(), outliers.min())
            else:
                max_val = data.max()
                min_val = data.min()

            hidden_outliers_times_hight = hidden_outliers_times[hidden_outliers > med]
            if len(hidden_outliers_times_hight) > 0:
                pp.ax.plot(hidden_outliers_times_hight, [max_val] * len(hidden_outliers_times_hight),
                           pp.style.super_outlier_point_shape_up, color=pp.colors.super_outlier_color,
                           label="{}Q+ hight Outliers".format(pp.style.outliers_hide_q_nd))

            hidden_outliers_times_low = hidden_outliers_times[hidden_outliers < med]
            if len(hidden_outliers_times_low) > 0:
                pp.ax.plot(hidden_outliers_times_low, [min_val] * len(hidden_outliers_times_low),
                           pp.style.super_outlier_point_shape_down, color=pp.colors.super_outlier_color,
                           label="{}Q+ low Outliers".format(pp.style.outliers_hide_q_nd))

    has_negative_dev = False
    plus_minus = "\xb1"

    if plot_avg_dev and len(data) < pp.style.avg_range * 2:
        logger.warning("Array %r to small to plot average over %s points", pp.title, pp.style.avg_range)
    elif plot_avg_dev:
        avg_vals = moving_average(data, pp.style.avg_range)
        dev_vals = moving_dev(data, pp.style.avg_range)
        avg_times = moving_average(data_times, pp.style.avg_range)

        if (plot_points and pp.style.approx_average) or (not plot_points and pp.style.approx_average_no_points):
            avg_vals = approximate_curve(avg_times, avg_vals, avg_times, pp.style.curve_approx_level)
            dev_vals = approximate_curve(avg_times, dev_vals, avg_times, pp.style.curve_approx_level)

        pp.ax.plot(avg_times, avg_vals, c=pp.colors.suppl_color1, label="Average")

        low_vals_dev = avg_vals - dev_vals * pp.style.dev_range_x
        hight_vals_dev = avg_vals + dev_vals * pp.style.dev_range_x
        if (pp.style.dev_range_x - int(pp.style.dev_range_x)) < 0.01:
            pp.ax.plot(avg_times, low_vals_dev, c=pp.colors.suppl_color2,
                       label="{}{}*stdev".format(plus_minus, int(pp.style.dev_range_x)))
        else:
            pp.ax.plot(avg_times, low_vals_dev, c=pp.colors.suppl_color2,
                       label="{}{}*stdev".format(plus_minus, pp.style.dev_range_x))
        pp.ax.plot(avg_times, hight_vals_dev, c=pp.colors.suppl_color2)
        has_negative_dev = low_vals_dev.min() < 0

    pp.ax.set_xlim(-5, max(time_points) + 5)
    pp.ax.set_xlabel("Time, seconds from test begin")

    if plot_avg_dev:
        pp.ax.set_ylabel("{}. Average and {}stddev over {} points".format(units, plus_minus, pp.style.avg_range))
    else:
        pp.ax.set_ylabel(units)

    if has_negative_dev:
        pp.ax.set_ylim(bottom=0)


@provide_plot(eng=True, no_legend=True, grid='y', noadjust=True)
def plot_lat_over_time(pp: PlotParams, ts: TimeSeries) -> None:
    times = ts.times - min(ts.times)
    step = len(times) / pp.style.lat_samples
    points = [times[int(i * step + 0.5)] for i in range(pp.style.lat_samples)]
    points.append(times[-1])
    bounds = list(zip(points[:-1], points[1:]))
    agg_data = []
    positions = []
    labels = []

    for begin, end in bounds:
        agg_hist = ts.data[begin:end].sum(axis=0)

        if pp.style.violin_instead_of_box:
            # cut outliers
            idx1, idx2 = hist_outliers_perc(agg_hist, pp.style.outliers_lat)
            agg_hist = agg_hist[idx1:idx2]
            curr_bins_vals = ts.histo_bins[idx1:idx2]

            correct_coef = pp.style.violin_point_count / sum(agg_hist)
            if correct_coef > 1:
                correct_coef = 1
        else:
            curr_bins_vals = ts.histo_bins
            correct_coef = 1

        vals = numpy.empty(shape=[numpy.sum(agg_hist)], dtype='float32')
        cidx = 0

        non_zero, = agg_hist.nonzero()
        for pos in non_zero:
            count = int(agg_hist[pos] * correct_coef + 0.5)

            if count != 0:
                vals[cidx: cidx + count] = curr_bins_vals[pos]
                cidx += count

        agg_data.append(vals[:cidx])
        positions.append((end + begin) / 2)
        labels.append(str((end + begin) // 2))

    if pp.style.violin_instead_of_box:
        patches = pp.ax.violinplot(agg_data, positions=positions, showmeans=True, showmedians=True, widths=step / 2)
        patches['cmeans'].set_color("blue")
        patches['cmedians'].set_color("green")
        if pp.style.legend_for_eng:
            legend_location = "center left"
            legend_bbox_to_anchor = (1.03, 0.81)
            pp.ax.legend([patches['cmeans'], patches['cmedians']], ["mean", "median"],
                         loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        pp.ax.boxplot(agg_data, 0, '', positions=positions, labels=labels, widths=step / 4)

    pp.ax.set_xlim(min(times), max(times))
    pp.ax.set_xlabel("Time, seconds from test begin, sampled for ~{} seconds".format(int(step)))
    pp.fig.subplots_adjust(right=pp.style.subplot_adjust_r)


@provide_plot(eng=True, no_legend=True, noaxis=True, long_plot=True)
def plot_histo_heatmap(pp: PlotParams, ts: TimeSeries, ylabel: str, xlabel: str = "time, s") -> None:

    # only histogram-based ts can be plotted
    assert len(ts.data.shape) == 2

    # Find global outliers. As load is expected to be stable during one job
    # outliers range can be detected globally
    total_hist = ts.data.sum(axis=0)
    idx1, idx2 = hist_outliers_perc(total_hist,
                                    bounds_perc=pp.style.outliers_lat,
                                    min_bins_left=pp.style.hm_hist_bins_count)

    # merge outliers with most close non-outliers cell
    orig_data = ts.data[:, idx1:idx2].copy()
    if idx1 > 0:
        orig_data[:, 0] += ts.data[:, :idx1].sum(axis=1)

    if idx2 < ts.data.shape[1]:
        orig_data[:, -1] += ts.data[:, idx2:].sum(axis=1)

    bins_vals = ts.histo_bins[idx1:idx2]

    # rebin over X axis
    # aggregate some lines in ts.data to plot ~style.hm_x_slots x bins
    agg_idx = float(len(orig_data)) / pp.style.hm_x_slots
    if agg_idx >= 2:
        idxs = list(map(int, numpy.round(numpy.arange(0, len(orig_data) + 1, agg_idx))))
        assert len(idxs) > 1
        data = numpy.empty([len(idxs) - 1, orig_data.shape[1]], dtype=numpy.float32)  # type: List[numpy.ndarray]
        for idx, (sidx, eidx) in enumerate(zip(idxs[:-1], idxs[1:])):
            data[idx] = orig_data[sidx:eidx,:].sum(axis=0) / (eidx - sidx)
    else:
        data = orig_data

    # rebin over Y axis
    # =================
    # don't using rebin_histogram here, as we need apply same bins for many arrays
    step = (bins_vals[-1] - bins_vals[0]) / pp.style.hm_hist_bins_count
    new_bins_edges = numpy.arange(pp.style.hm_hist_bins_count) * step + bins_vals[0]
    bin_mapping = numpy.clip(numpy.searchsorted(new_bins_edges, bins_vals) - 1, 0, len(new_bins_edges) - 1)

    # map origin bins ranges to heatmap bins, iterate over rows
    cmap = []
    for line in data:
        curr_bins = [0] * pp.style.hm_hist_bins_count
        for idx, count in zip(bin_mapping, line):
            curr_bins[idx] += count
        cmap.append(curr_bins)
    ncmap = numpy.array(cmap)

    histo = ncmap.sum(axis=0).reshape((-1,))
    ax, _ = do_plot_hmap_with_histo(pp.fig, ncmap, histo, new_bins_edges,
                                    cmap=pp.colors.hmap_cmap,
                                    cbar=pp.style.heatmap_colorbar, avg_labels=True)
    ax.set(ylabel=ylabel, xlabel=xlabel)

