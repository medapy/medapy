"""
Metric prefix conversion utilities.

This module provides functions for converting between numeric values and SI prefixes,
including automatic prefix selection, formatting, and parsing of prefixed values.
"""

import math

# SI prefixes dictionary (name, symbol, multiplier)
# Range: 1e-24 (yocto) to 1e24 (yotta)
SI_PREFIXES = {
    'yotta': ('Y', 1e24),
    'zetta': ('Z', 1e21),
    'exa':   ('E', 1e18),
    'peta':  ('P', 1e15),
    'tera':  ('T', 1e12),
    'giga':  ('G', 1e9),
    'mega':  ('M', 1e6),
    'kilo':  ('k', 1e3),
    'milli': ('m', 1e-3),
    'micro': ('u', 1e-6),   # 'u' for filename compatibility
    'nano':  ('n', 1e-9),
    'pico':  ('p', 1e-12),
    'femto': ('f', 1e-15),
    'atto':  ('a', 1e-18),
    'zepto': ('z', 1e-21),
    'yocto': ('y', 1e-24),
}

# Auto-generate lookup dictionaries
SYMBOL_TO_MULTIPLIER = {symbol: multiplier for symbol, multiplier in SI_PREFIXES.values()}
MULTIPLIER_TO_SYMBOL = {multiplier: symbol for symbol, multiplier in SI_PREFIXES.values()}

# Sorted list of multipliers for finding nearest prefix
_SORTED_MULTIPLIERS = sorted(MULTIPLIER_TO_SYMBOL.keys(), reverse=True)


def prefix_to_multiplier(prefix: str) -> float:
    """
    Convert a metric prefix to its numeric multiplier.

    Parameters
    ----------
    prefix : str
        SI prefix symbol (e.g., 'M', 'k', 'm', 'u')

    Returns
    -------
    float
        The numeric multiplier corresponding to the prefix

    Raises
    ------
    ValueError
        If the prefix is not recognized

    Examples
    --------
    >>> prefix_to_multiplier('M')
    1000000.0
    >>> prefix_to_multiplier('k')
    1000.0
    >>> prefix_to_multiplier('m')
    0.001
    >>> prefix_to_multiplier('u')
    1e-06
    """
    if prefix in SYMBOL_TO_MULTIPLIER:
        return SYMBOL_TO_MULTIPLIER[prefix]

    raise ValueError(
        f"Unknown prefix: '{prefix}'. "
        f"Valid prefixes: {list(SYMBOL_TO_MULTIPLIER.keys())}"
    )


def multiplier_to_prefix(multiplier: float) -> str:
    """
    Convert a numeric multiplier to its corresponding SI prefix (exact match only).

    Parameters
    ----------
    multiplier : float
        The numeric multiplier (must be exact: 1e12, 1e9, 1e6, 1e3, 1e-3, 1e-6, 1e-9, 1e-12)

    Returns
    -------
    str
        The SI prefix symbol

    Raises
    ------
    ValueError
        If the multiplier doesn't correspond to a standard SI prefix.
        Use find_best_prefix() for non-standard multipliers.

    Examples
    --------
    >>> multiplier_to_prefix(1e6)
    'M'
    >>> multiplier_to_prefix(1e3)
    'k'
    >>> multiplier_to_prefix(1e-3)
    'm'
    >>> multiplier_to_prefix(1e-6)
    'u'
    """
    if multiplier in MULTIPLIER_TO_SYMBOL:
        return MULTIPLIER_TO_SYMBOL[multiplier]

    raise ValueError(
        f"Multiplier {multiplier} doesn't correspond to a standard SI prefix. "
        f"Valid multipliers: {sorted(MULTIPLIER_TO_SYMBOL.keys())}. "
        f"Use find_best_prefix() for non-standard multipliers."
    )


def find_best_prefix(multiplier: float) -> str:
    """
    Find the closest valid SI prefix for any multiplier (including non-standard ones).

    Parameters
    ----------
    multiplier : float
        The numeric multiplier (e.g., 1e7, 1e4, 1e-4)

    Returns
    -------
    str
        The closest SI prefix symbol

    Examples
    --------
    >>> find_best_prefix(1e7)
    'M'
    >>> find_best_prefix(1e4)
    'k'
    >>> find_best_prefix(1e-4)
    'm'
    >>> find_best_prefix(1e6)
    'M'
    """
    if multiplier <= 0:
        raise ValueError(f"Multiplier must be positive, got {multiplier}")

    # If exact match exists, return it
    if multiplier in MULTIPLIER_TO_SYMBOL:
        return MULTIPLIER_TO_SYMBOL[multiplier]

    # Find closest standard multiplier on log scale
    log_mult = math.log10(multiplier)
    closest_multiplier = min(
        _SORTED_MULTIPLIERS,
        key=lambda m: abs(math.log10(m) - log_mult)
    )

    return MULTIPLIER_TO_SYMBOL[closest_multiplier]


def value_to_prefix(
    value: float,
    precision: int | str | None = 'auto',
    measurement_precision: str | float | None = None,
    use_exp_fallback: bool = True
) -> tuple[float, str]:
    """
    Convert a numeric value to a scaled value with appropriate SI prefix.

    Automatically selects prefix to scale values:
    - Values >= 1000 or < 1 get a prefix
    - Values in [1, 1000) get no prefix
    - Values outside SI prefix range return unscaled (empty prefix) if use_exp_fallback=True

    Parameters
    ----------
    value : float
        The numeric value to convert
    precision : int, 'auto', or None, optional
        Decimal places for output (default is 'auto'):
        - int: exact decimal places (e.g., 2 → "1.50")
        - 'auto' or None: minimum needed decimals (e.g., 100.0 → "100", 100.5 → "100.5")
    measurement_precision : str or float, optional
        Measurement resolution for rounding before scaling:
        - str: '<value> <prefix>' format (e.g., '1 m', '10 u')
        - float: direct value (e.g., 0.001)
        Use this to avoid false precision in output.
    use_exp_fallback : bool, optional
        If True, return unscaled value with empty prefix for values outside prefix range.
        If False, clamp to nearest available prefix. Default is True.

    Returns
    -------
    tuple[float, str]
        A tuple of (scaled_value, prefix_symbol)
        For values outside SI range with fallback: (original_value, '')

    Examples
    --------
    >>> value_to_prefix(1500000)
    (1.5, 'M')
    >>> value_to_prefix(100.0)
    (100, '')
    >>> value_to_prefix(100.0, precision=2)
    (100.0, '')
    >>> value_to_prefix(0.1417, measurement_precision='1 m')
    (142, 'm')
    >>> value_to_prefix(1e27)  # Outside prefix range
    (1e+27, '')
    >>> value_to_prefix(1e-27)  # Outside prefix range
    (1e-27, '')
    """
    # Apply measurement precision rounding first (if specified)
    if measurement_precision is not None:
        value = _round_to_measurement_precision(value, measurement_precision)

    if value == 0:
        return (0.0, '')

    abs_value = abs(value)

    # No prefix needed for values in [1, 1000)
    if 1 <= abs_value < 1000:
        if precision == 'auto' or precision is None:
            precision = _detect_min_precision(value)
        if precision is not None:
            value = round(value, precision)
        return (value, '')

    # Find appropriate prefix based on order of magnitude
    exponent = math.floor(math.log10(abs_value))

    # Round to nearest multiple of 3 for SI prefixes (use integer division)
    prefix_exponent = (exponent // 3) * 3

    # Check if outside available range [-24, 24]
    if use_exp_fallback and (prefix_exponent < -24 or prefix_exponent > 24):
        # Return unscaled value with empty prefix (will use exponential notation in formatting)
        return (value, '')

    # Clamp to available range [-24, 24] if not using fallback
    prefix_exponent = max(-24, min(24, prefix_exponent))
    prefix_multiplier = 10 ** prefix_exponent

    # Find the best matching prefix
    prefix_symbol = find_best_prefix(prefix_multiplier)
    actual_multiplier = prefix_to_multiplier(prefix_symbol)

    # Scale the value
    scaled_value = value / actual_multiplier

    # Determine precision
    if precision == 'auto' or precision is None:
        precision = _detect_min_precision(scaled_value)

    # Round if precision is specified
    if precision is not None:
        scaled_value = round(scaled_value, precision)

    return (scaled_value, prefix_symbol)


def format_with_prefix(
    value: float,
    precision: int | str | None = 'auto',
    measurement_precision: str | float | None = None,
    use_exp_fallback: bool = True,
    use_space: bool = True
) -> str:
    """
    Format a numeric value with appropriate SI prefix.

    Parameters
    ----------
    value : float
        The numeric value to format
    precision : int, 'auto', or None, optional
        Decimal places for output (default is 'auto'):
        - int: exact decimal places
        - 'auto' or None: minimum needed decimals
    measurement_precision : str or float, optional
        Measurement resolution (see value_to_prefix for details)
    use_exp_fallback : bool, optional
        If True, use exponential notation for values outside prefix range.
        If False, clamp to nearest available prefix. Default is True.
    use_space : bool, optional
        If True, include space between value and prefix (e.g., '1.5 M').
        If False, no space (e.g., '1.5M') - useful for filenames.
        Default is True.

    Returns
    -------
    str
        Formatted string with value and prefix

    Examples
    --------
    >>> format_with_prefix(1500000)
    '1.5 M'
    >>> format_with_prefix(1500000, use_space=False)
    '1.5M'
    >>> format_with_prefix(100.0)
    '100'
    >>> format_with_prefix(100.0, precision=2)
    '100.00'
    >>> format_with_prefix(0.1417, measurement_precision='1 m')
    '142 m'
    >>> format_with_prefix(0.1417, measurement_precision='1 m', use_space=False)
    '142m'
    >>> format_with_prefix(1e27)
    '1e+27'
    >>> format_with_prefix(1e-27)
    '1e-27'
    """
    scaled_value, prefix = value_to_prefix(
        value,
        precision=precision,
        measurement_precision=measurement_precision,
        use_exp_fallback=use_exp_fallback
    )

    # Check if we need exponential notation (large/small value with no prefix)
    abs_value = abs(scaled_value)
    needs_exp_notation = (
        prefix == '' and
        abs_value != 0 and
        (abs_value >= 1e24 or abs_value < 1e-24)
    )

    if needs_exp_notation:
        # Use exponential notation
        if precision == 'auto' or precision is None:
            # Let Python decide precision with 'g' format
            return f"{scaled_value:g}"
        else:
            # Use explicit precision with 'e' format
            return f"{scaled_value:.{precision}e}"

    # Standard formatting with SI prefix
    if precision == 'auto' or precision is None:
        precision = _detect_min_precision(scaled_value)

    if prefix:
        space = ' ' if use_space else ''
        return f"{scaled_value:.{precision}f}{space}{prefix}"
    else:
        return f"{scaled_value:.{precision}f}"


def parse_prefixed_value(value_str: str) -> float:
    """
    Parse a string with optional SI prefix to numeric value.

    Supports formats with or without spaces between number and prefix.

    Parameters
    ----------
    value_str : str
        String representation with optional SI prefix.
        Formats: "100 m", "100m", "1.5 M", "1.5e3 k", "100"

    Returns
    -------
    float
        Numeric value with prefix applied

    Raises
    ------
    ValueError
        If format is invalid or prefix is not recognized

    Examples
    --------
    >>> parse_prefixed_value("100 m")
    0.1
    >>> parse_prefixed_value("100m")
    0.1
    >>> parse_prefixed_value("1.5 M")
    1500000.0
    >>> parse_prefixed_value("1.5e3 k")
    1500000.0
    >>> parse_prefixed_value("100")
    100.0
    >>> parse_prefixed_value("-50 u")
    -5e-05
    """
    # Remove all whitespace
    value_str = value_str.strip().replace(' ', '')

    if not value_str:
        raise ValueError("Empty string cannot be parsed")

    # Try to extract prefix (last character if it's a valid prefix)
    prefix = ''
    numeric_part = value_str
    potential_prefix = value_str[-1]

    if potential_prefix in SYMBOL_TO_MULTIPLIER:
        prefix = potential_prefix
        numeric_part = value_str[:-1]
    elif potential_prefix.isalpha():
        # Last character is a letter but not a valid prefix
        raise ValueError(
            f"Unknown prefix: '{potential_prefix}'. "
            f"Valid prefixes: {list(SYMBOL_TO_MULTIPLIER.keys())}"
        )

    # Parse the numeric part
    try:
        value = float(numeric_part)
    except ValueError:
        raise ValueError(
            f"Cannot parse numeric value from '{numeric_part}'. "
            f"Expected format: '<number> <prefix>' or '<number><prefix>'"
        )

    # Apply prefix multiplier if present
    if prefix:
        multiplier = prefix_to_multiplier(prefix)
        value *= multiplier

    return value
# Helper functions for smart precision

def _detect_min_precision(value: float, max_decimals: int = 3) -> int:
    """
    Detect the minimum number of decimal places needed to represent a value.

    Parameters
    ----------
    value : float
        The value to analyze
    max_decimals : int, optional
        Maximum decimal places to check (default is 3)

    Returns
    -------
    int
        Minimum decimal places needed

    Examples
    --------
    >>> _detect_min_precision(100.0)
    0
    >>> _detect_min_precision(100.5)
    1
    >>> _detect_min_precision(100.123)
    3
    """
    for decimals in range(max_decimals + 1):
        # Check if rounding to this many decimals preserves the value
        # (within floating point tolerance)
        if round(value, decimals) == round(value, max_decimals):
            return decimals
    return max_decimals


def _parse_measurement_precision(prec_str: str) -> float:
    """
    Parse measurement precision string to numeric value.

    Parameters
    ----------
    prec_str : str
        Measurement precision in format '<number> <prefix>'
        Examples: '1 m', '10 u', '0.5 k'

    Returns
    -------
    float
        The numeric value of the measurement precision

    Raises
    ------
    ValueError
        If the format is invalid or prefix is not recognized

    Examples
    --------
    >>> _parse_measurement_precision('1 m')
    0.001
    >>> _parse_measurement_precision('10 u')
    1e-05
    >>> _parse_measurement_precision('0.5 k')
    500.0
    """
    parts = prec_str.strip().split()
    if len(parts) != 2:
        raise ValueError(
            f"Measurement precision format should be '<number> <prefix>', "
            f"got '{prec_str}'"
        )

    try:
        magnitude = float(parts[0])
    except ValueError:
        raise ValueError(
            f"Invalid number in measurement precision: '{parts[0]}'"
        )

    prefix = parts[1]
    multiplier = prefix_to_multiplier(prefix)  # Will raise if prefix is invalid

    return magnitude * multiplier


def _round_to_measurement_precision(
    value: float,
    meas_prec: str | float
) -> float:
    """
    Round value to the specified measurement precision.

    Parameters
    ----------
    value : float
        The value to round
    meas_prec : str or float
        Measurement precision:
        - str: '<number> <prefix>' format (e.g., '1 m')
        - float: direct precision value (e.g., 0.001)

    Returns
    -------
    float
        Value rounded to measurement precision

    Raises
    ------
    ValueError
        If measurement precision is not positive or format is invalid

    Examples
    --------
    >>> _round_to_measurement_precision(0.1417, '1 m')
    0.142
    >>> _round_to_measurement_precision(0.1417, 0.001)
    0.142
    >>> _round_to_measurement_precision(1417, '10')
    1420.0
    """
    if isinstance(meas_prec, str):
        meas_prec = _parse_measurement_precision(meas_prec)

    if meas_prec <= 0:
        raise ValueError(
            f"Measurement precision must be positive, got {meas_prec}"
        )

    # Round to nearest multiple of meas_prec
    return round(value / meas_prec) * meas_prec
