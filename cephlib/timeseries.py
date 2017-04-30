from typing import List, Tuple


import numpy


def approximate_ts(times: List[float], values: List[float], begin: float, end: float,
                   step: float = 1000000) -> List[float]:
    if len(times) != len(values):
        raise AssertionError("Times and values arrays has different sizes")

    if begin < times[0] or end > times[-1] or end <= begin:
        raise AssertionError("Can't approximate as at least one border is not beelong data range or incorect borders")

    pos1, pos2 = numpy.searchsorted(times, (begin, end))

    # current real data time chunk begin time
    edge_it = iter(times[pos1 - 1: pos2 + 1])

    # current real data value
    val_it = iter(values[pos1 - 1: pos2 + 1])

    # result array, cumulative value per second
    result = numpy.zeros(int(end - begin) // step)
    idx = 0
    curr_summ = 0

    # end of current time slot
    results_cell_ends = begin + step

    # hack to unify looping
    real_data_end = next(edge_it)
    while results_cell_ends <= end:
        real_data_start = real_data_end
        real_data_end = next(edge_it)
        real_val_left = next(val_it)

        # real data "speed" for interval [real_data_start, real_data_end]
        real_val_ps = float(real_val_left) / (real_data_end - real_data_start)

        while real_data_end >= results_cell_ends and results_cell_ends <= end:
            # part of current real value, which is fit into current result cell
            curr_real_chunk = int((results_cell_ends - real_data_start) * real_val_ps)

            # calculate rest of real data for next result cell
            real_val_left -= curr_real_chunk
            result[idx] = curr_summ + curr_real_chunk
            idx += 1
            curr_summ = 0

            # adjust real data start time
            real_data_start = results_cell_ends
            results_cell_ends += step

        # don't lost any real data
        curr_summ += real_val_left

    return result


# data is timeseries of 1D arrays, each array is view on system parts load at come time
# E.G. OSD loads at t0. t0 + 1, t0 + 2, ...
# return 2D heatmap array
def prepare_heatmap(data: List[List[float]],
                    bins_vals: List[int],
                    bins_count: int,
                    outliers_perc: Tuple[float, float]) -> Tuple[List[List[float]], List[float]]:
    """
    :param data: list of histograms, one per line
    :param bins_vals: values at center of each bin
    :param bins_count: result bin count for each column
    :param outliers_perc: pair of outliers limits tupically (0.25, 0.75)
    :return:
    """

    assert len(data.shape) == 2
    assert data.shape[1] == len(bins_vals)

    total_hist = data.sum(axis=0)

    # idx1, idx2 = hist_outliers_perc(total_hist, style.outliers_lat)
    idx1, idx2 = ts_hist_outliers_perc(data, bounds_perc=outliers_perc)

    # don't cut too many bins
    min_bins_left = bins_count
    if idx2 - idx1 < min_bins_left:
        missed = min_bins_left - (idx2 - idx1) // 2
        idx2 = min(len(total_hist), idx2 + missed)
        idx1 = max(0, idx1 - missed)

    data = data[:, idx1:idx2]
    bins_vals = bins_vals[idx1:idx2]

    # don't using rebin_histogram here, as we need apply same bins for many arrays
    step = (bins_vals[-1] - bins_vals[0]) / bins_count
    new_bins_edges = numpy.arange(bins_count) * step + bins_vals[0]
    bin_mapping = numpy.clip(numpy.searchsorted(new_bins_edges, bins_vals) - 1, 0, len(new_bins_edges) - 1)

    # map origin bins ranges to heatmap bins, iterate over rows
    cmap = []
    for line in data:
        curr_bins = [0] * bins_count
        for idx, count in zip(bin_mapping, line):
            curr_bins[idx] += count
        cmap.append(curr_bins)

    return numpy.array(cmap), new_bins_edges


