"""
Deprecated alias for the modules that replaced ``medapy.utils.misc``.

The array/DataFrame processing functions now live in ``medapy.analysis.processing``.
The two validators shared with the analysis code were promoted to public names in
``medapy.utils.validations``: ``_validate_option`` -> ``validate_option`` and
``_validate_xy`` -> ``validate_xy``.

Attribute lookups are forwarded to the new locations with a ``DeprecationWarning``.
This module will be removed in a future release.
"""
import warnings

from medapy.analysis import processing as _processing
from medapy.utils import validations as _validations

# Old private names -> promoted public names in utils.validations
_MOVED_VALIDATORS = {
    '_validate_option': ('medapy.utils.validations.validate_option', _validations.validate_option),
    '_validate_xy': ('medapy.utils.validations.validate_xy', _validations.validate_xy),
}


def __getattr__(name: str):
    if name in _MOVED_VALIDATORS:
        new_path, target = _MOVED_VALIDATORS[name]
    else:
        try:
            target = getattr(_processing, name)
        except AttributeError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None
        new_path = f'medapy.analysis.processing.{name}'

    warnings.warn(
        f"'medapy.utils.misc.{name}' is deprecated; use '{new_path}' instead. "
        "'medapy.utils.misc' will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return target


def __dir__() -> list[str]:
    return sorted(set(dir(_processing)) | set(_MOVED_VALIDATORS))
