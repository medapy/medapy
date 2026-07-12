import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """
    Import an optional dependency, or raise a helpful error.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``'scipy.signal'``.
    extra : str
        Name of the packaging extra that provides the module, used to build
        the installation hint in the error message.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed.

    Examples
    --------
    >>> signal = require('scipy.signal', extra='fit')
    >>> signal.savgol_filter([1, 2, 3, 4, 5], 3, 1)  # doctest: +SKIP
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{module}' is required for this function but is not installed. "
            f"Install it with: pip install medapy[{extra}]"
        ) from exc
