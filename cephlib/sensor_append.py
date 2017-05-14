import array
from typing import Tuple, Any, List

from .istorage import IStorage
from .types import DataSource, NumVector

try:
    import numpy
except ImportError:
    numpy = None


