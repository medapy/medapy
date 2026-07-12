import numbers

import numpy as np
import numpy.typing as npt


def class_in_iterable(iterable, class_obj, iter_name):
    if not all(isinstance(item, class_obj) for item in iterable):
        raise TypeError(
            f"All items in {iter_name} must be {class_obj.__name__} objects"
        )


def validate_value_range(val_range):
    """
    Validate a numeric range for filtering.

    Args:
        val_range: Tuple (min, max) where either value can be None for open bounds

    Returns:
        tuple: (left, right) preserving original numeric types, None replaced with inf

    Raises:
        TypeError: If val_range is not iterable or values are non-numeric
        ValueError: If val_range doesn't have exactly 2 values

    Examples:
        >>> validate_value_range((5, 10))
        (5, 10)
        >>> validate_value_range((5.0, None))
        (5.0, inf)
        >>> validate_value_range((None, 10))
        (-inf, 10)
    """
    try:
        range_list = list(val_range)
    except TypeError:
        raise TypeError("Range must be an iterable (tuple, list, etc.)")

    if len(range_list) != 2:
        raise ValueError(f"Range must contain exactly 2 values, got {len(range_list)}")

    left, right = range_list

    # Validate and handle None for left bound
    if left is None:
        left = float("-inf")
    elif not isinstance(left, numbers.Number):
        raise TypeError(
            f"Left bound must be numeric or None, got {type(left).__name__}"
        )

    # Validate and handle None for right bound
    if right is None:
        right = float("inf")
    elif not isinstance(right, numbers.Number):
        raise TypeError(
            f"Right bound must be numeric or None, got {type(right).__name__}"
        )

    # Left is always minimum, swap if needed
    if left > right:
        left, right = right, left

    return (left, right)


def validate_option(value, allowed_values, param_name):
    """
    Validate if a value is among allowed options.

    Parameters
    ----------
    value : Any
        The value to validate
    allowed_values : set or tuple or list
        Collection of allowed values
    param_name : str
        Name of the parameter being validated, used in error message

    Returns
    -------
    Any
        The validated value

    Raises
    ------
    ValueError
        If value is not in allowed_values

    Examples
    --------
    >>> validate_option('exclude', ['exclude', 'raise'], 'handle_na')
    'exclude'
    >>> validate_option('invalid', ['exclude', 'raise'], 'handle_na')
    ValueError: handle_na must be one of: 'exclude', 'raise'
    """
    if value not in allowed_values:
        options_str = "', '".join(str(v) for v in allowed_values)
        raise ValueError(f"'{param_name}' must be one of: '{options_str}'")
    return value


def validate_xy(x: npt.ArrayLike, y: npt.ArrayLike, handle_na: str = 'raise') -> tuple[np.ndarray, np.ndarray]:
    """Validate and preprocess x and y input arrays.

    Parameters
    ----------
    x : array_like
        Independent variable values
    y : array_like
        Dependent variable values
    handle_na : str, default 'raise'
        How to handle NaN/inf values: raises error if 'raise`,
        excludes them from arrays if 'exclude'

    Returns
    -------
    x_clean, y_clean : ndarray
        Validated and cleaned input arrays

    Raises
    ------
    ValueError
        If inputs have invalid shapes or contain NaN/inf with handle_na='raise'
    TypeError
        If inputs cannot be converted to numpy arrays
    """
    # Convert inputs to arrays and validate
    x, y = np.asarray(x), np.asarray(y)

    mask_na = np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y)
    if mask_na.any():
        n_invalid = np.sum(mask_na)
        if handle_na == 'raise':
            raise ValueError(f"Found {n_invalid} NaN/Inf values in data")
        # handle_na == 'exclude'
        x, y = x[~mask_na], y[~mask_na]

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")

    if x.ndim != 1:
        raise ValueError("x and y must be 1-dimensional")
    return x, y
