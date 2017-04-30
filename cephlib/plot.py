import numpy
import warnings

from matplotlib import gridspec, ticker
from matplotlib import pyplot as plt
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from .common import float2str
from .numeric import auto_edges


def hmap_from_2d(data, max_xbins=25, noval=None):
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

    # calculate how many 'data' rows fit into single output interval
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


def process_heatmap_data(values, bin_ranges, cut_percentile=(0.02, 0.98), ybins=20, log_edges=True, bins=None):
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


def plot_heatmap(ax, hmap_vals, chunk_ranges, bins=None):
    assert len(hmap_vals.shape) == 1
    heatmap, bins = process_heatmap_data(hmap_vals, chunk_ranges, bins=bins)
    labels = list(map(float2str, bins))
    seaborn.heatmap(heatmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(labels))))
    ax.set_yticklabels(labels, rotation='horizontal')
    return bins


def plot_histo(ax, vals, bins=None, kde=False, left=None, right=None):
    assert len(vals.shape) == 1
    seaborn.distplot(vals, bins=bins, ax=ax, kde=kde)
    ax.set_yticklabels([])

    if left is not None or right is not None:
        ax.set_xlim(left=left, right=right)


def plot_hmap_with_y_histo(fig, data, chunk_ranges, boxes=3, kde=False, bins=None):
    assert len(data.shape) == 1

    gs = gridspec.GridSpec(1, boxes)
    ax = fig.add_subplot(gs[0, :boxes - 1])

    bins = plot_heatmap(ax, data, chunk_ranges, bins=bins)

    ax2 = fig.add_subplot(gs[0, boxes - 1])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_ylim(top=len(bins) - 1, bottom=0)
    # seaborn.distplot(data, bins=bins, ax=ax2, kde=kde, vertical=True)

    bins_populations, _ = numpy.histogram(data, bins)
    ax2.barh(numpy.arange(len(bins_populations)) + 0.5, width=bins_populations)

    return ax, ax2
