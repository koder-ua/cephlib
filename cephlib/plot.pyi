import numpy
import matplotlib.axes
from matplotlib.figure import Figure
from typing import Any, List, Tuple


ndarray2d = numpy.ndarray

def hmap_from_2d(data: numpy.ndarray, max_xbins: int = 25, noval: Any = None) -> numpy.ndarray: ...

def process_heatmap_data(values: numpy.ndarray,
                         bin_ranges: List[Tuple[float, float]],
                         cut_percentile: Tuple[float, float] = (0.02, 0.98),
                         ybins: int = 20,
                         log_edges: bool = True) -> numpy.ndarray: ...

def plot_heatmap(ax: matplotlib.axes.Axes,
                 hmap_vals: numpy.ndarray,
                 chunk_ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]: ...

def plot_histo(ax: matplotlib.axes.Axes,
               vals: numpy.ndarray, bins: List[float] = None, kde: bool = False,
               left: float = None, right: float = None) -> None: ...

def plot_hmap_with_y_histo(fig: Figure,
                           data: numpy.ndarray,
                           chunk_ranges: List[Tuple[float, float]],
                           boxes: int = 3,
                           kde: bool = False,
                           bins: List[float] = None) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes] : ...
