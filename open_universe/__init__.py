try:
    from ._version import __version__
except ImportError:
    __version__ = "nightly"
from . import datasets, inference_utils, layers, lora, losses, metrics, networks, utils
