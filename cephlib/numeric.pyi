from typing import List

import numpy


ndarray1d = numpy.ndarray


def auto_edges(vals: ndarray1d, log_base: float = 2, bins: int = 20,
               round_base: int = 10, log_space: bool = True) -> ndarray1d: ...

def auto_edges2(lower: float, upper: float, log_base: float = 2,
                bins: int = 20, round_base: int = 10, log_space: bool = True) -> ndarray1d: ...
