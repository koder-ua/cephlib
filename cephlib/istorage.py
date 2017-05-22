import abc
from typing import Tuple

# extra imports, as this file provides common API for other modules
from .istorage_nnp import (IStorable, Storable, ISerializer, IStorageNNP, PathSelector,
                           ISensorStorageNNP, IImagesStorage, ISimpleStorage, _Raise, ObjClass, SensorsIter)

from .types import DataSource
from .numeric_types import TimeSeries, ArrayData


class IStorage(IStorageNNP, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_array(self, path: str) -> ArrayData:
        pass

class ISensorStorage(ISensorStorageNNP, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_sensor(self, ds: DataSource, time_range: Tuple[float, float]) -> TimeSeries:
        pass
