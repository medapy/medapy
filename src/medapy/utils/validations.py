import numbers


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
