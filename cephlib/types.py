import array
from typing import TypeVar, List, Union

try:
    import numpy
    NumVector = Union[numpy.ndarray, array.array, List[int], List[float]]
except ImportError:
    NumVector = Union[array.array, List[int], List[float]]


TNumber = TypeVar('TNumber', int, float)
Number = Union[int, float]

array1d = NumVector
array2d = NumVector
