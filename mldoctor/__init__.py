"""ML Doctor: static analysis, runtime monitoring, and ML training diagnostics."""

__version__ = "0.3.0"
VERSION = __version__

from .models import Diagnostic, Severity
from .mldoctor import main

__all__ = ["Diagnostic", "Severity", "VERSION", "__version__", "main"]
