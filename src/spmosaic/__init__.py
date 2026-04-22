"""Top-level package for spMosaic."""

__version__ = "0.1.0"

from .smoothing import gene_smooth
from .domains import domain_detection
from .utils import get_rscript_path, get_r_version

__all__ = ["gene_smooth", "domain_detection", "get_rscript_path", "get_r_version"]
