import importlib

from . import processing
from . import proc_pandas

# Domain packages are imported on first attribute access so that their optional
# dependencies (e.g. lmfit) are not required to import medapy.analysis
_LAZY_DOMAINS = {'etr': 'medapy.analysis.electron_transport'}


def __getattr__(name: str):
    if name not in _LAZY_DOMAINS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    # Importing the domain package also registers its pandas accessor
    return getattr(importlib.import_module(_LAZY_DOMAINS[name]), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_DOMAINS))


__all__ = ['processing', 'proc_pandas', 'etr']
