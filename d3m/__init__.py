from .models import ConvModel, MLPModel, ResNetModel, BERTModel
from .monitors import D3MBayesianMonitor, D3MFullInformationMonitor
from ._metadata import __author__, __description__, __email__, __license__, __url__
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("d3m")  
except PackageNotFoundError:
    __version__ = "unknown"
