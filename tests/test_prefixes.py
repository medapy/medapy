"""
Tests for metric prefix conversion utilities.

This module tests the SI prefix conversion functions in medapy.utils.prefixes,
including prefix-to-multiplier conversions, value formatting, and parsing.
"""

import pytest
import math
from medapy.utils.prefixes import (
    prefix_to_multiplier,
    multiplier_to_prefix,
    find_best_prefix,
    value_to_prefix,
    format_with_prefix,
    parse_prefixed_value,
    detect_min_precision,
    _round_to_measurement_precision,
    SYMBOL_TO_MULTIPLIER,
    MULTIPLIER_TO_SYMBOL,
)


class TestPrefixToMultiplier:
    """Tests for prefix_to_multiplier function."""

    def test_standard_prefixes(self):
        """Test conversion of standard SI prefixes to multipliers."""
        assert prefix_to_multiplier('Y') == 1e24
        assert prefix_to_multiplier('Z') == 1e21
        assert prefix_to_multiplier('E') == 1e18
        assert prefix_to_multiplier('P') == 1e15
        assert prefix_to_multiplier('T') == 1e12
        assert prefix_to_multiplier('G') == 1e9
        assert prefix_to_multiplier('M') == 1e6
        assert prefix_to_multiplier('k') == 1e3
        assert prefix_to_multiplier('m') == 1e-3
        assert prefix_to_multiplier('u') == 1e-6
        assert prefix_to_multiplier('n') == 1e-9
        assert prefix_to_multiplier('p') == 1e-12
        assert prefix_to_multiplier('f') == 1e-15
        assert prefix_to_multiplier('a') == 1e-18
        assert prefix_to_multiplier('z') == 1e-21
        assert prefix_to_multiplier('y') == 1e-24

    def test_invalid_prefix(self):
        """Test that invalid prefixes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown prefix"):
            prefix_to_multiplier('X')

        with pytest.raises(ValueError, match="Unknown prefix"):
            prefix_to_multiplier('μ')  # Should use 'u' not 'μ'

        with pytest.raises(ValueError, match="Unknown prefix"):
            prefix_to_multiplier('')


class TestMultiplierToPrefix:
    """Tests for multiplier_to_prefix function."""

    def test_standard_multipliers(self):
        """Test conversion of standard multipliers to prefixes."""
        assert multiplier_to_prefix(1e24) == 'Y'
        assert multiplier_to_prefix(1e21) == 'Z'
        assert multiplier_to_prefix(1e18) == 'E'
        assert multiplier_to_prefix(1e15) == 'P'
        assert multiplier_to_prefix(1e12) == 'T'
        assert multiplier_to_prefix(1e9) == 'G'
        assert multiplier_to_prefix(1e6) == 'M'
        assert multiplier_to_prefix(1e3) == 'k'
        assert multiplier_to_prefix(1e-3) == 'm'
        assert multiplier_to_prefix(1e-6) == 'u'
        assert multiplier_to_prefix(1e-9) == 'n'
        assert multiplier_to_prefix(1e-12) == 'p'
        assert multiplier_to_prefix(1e-15) == 'f'
        assert multiplier_to_prefix(1e-18) == 'a'
        assert multiplier_to_prefix(1e-21) == 'z'
        assert multiplier_to_prefix(1e-24) == 'y'

    def test_non_standard_multipliers(self):
        """Test that non-standard multipliers raise ValueError."""
        with pytest.raises(ValueError, match="doesn't correspond to a standard SI prefix"):
            multiplier_to_prefix(1e4)

        with pytest.raises(ValueError, match="Use find_best_prefix"):
            multiplier_to_prefix(1e7)

        with pytest.raises(ValueError, match="doesn't correspond to a standard SI prefix"):
            multiplier_to_prefix(1e-4)


class TestFindBestPrefix:
    """Tests for find_best_prefix function."""

    def test_exact_matches(self):
        """Test that exact multipliers return correct prefixes."""
        assert find_best_prefix(1e6) == 'M'
        assert find_best_prefix(1e3) == 'k'
        assert find_best_prefix(1e-3) == 'm'
        assert find_best_prefix(1e-6) == 'u'

    def test_nearest_prefix_selection(self):
        """Test finding nearest prefix for non-standard multipliers."""
        # 1e7 is closer to 1e6 (M) than 1e9 (G) on log scale
        assert find_best_prefix(1e7) == 'M'

        # 1e4 is closer to 1e3 (k) than 1e6 (M)
        assert find_best_prefix(1e4) == 'k'

        # 1e-4 is closer to 1e-3 (m) than 1e-6 (u)
        assert find_best_prefix(1e-4) == 'm'

        # Test edge cases
        assert find_best_prefix(1e5) == 'M'  # Closer to mega
        assert find_best_prefix(1e2) == 'k'  # Closer to kilo

    def test_invalid_multipliers(self):
        """Test that invalid multipliers raise ValueError."""
        with pytest.raises(ValueError, match="Multiplier must be positive"):
            find_best_prefix(0)

        with pytest.raises(ValueError, match="Multiplier must be positive"):
            find_best_prefix(-1e3)


class TestValueToPrefix:
    """Tests for value_to_prefix function."""

    def test_basic_conversions(self):
        """Test basic value to prefix conversions."""
        # Values that need prefixes
        scaled, prefix = value_to_prefix(1500000)
        assert scaled == 1.5
        assert prefix == 'M'

        scaled, prefix = value_to_prefix(0.001)
        assert scaled == 1
        assert prefix == 'm'

        scaled, prefix = value_to_prefix(1500)
        assert scaled == 1.5
        assert prefix == 'k'

    def test_no_prefix_needed(self):
        """Test values that don't need prefixes (1 <= value < 1000)."""
        scaled, prefix = value_to_prefix(100.0)
        assert scaled == 100
        assert prefix == ''

        scaled, prefix = value_to_prefix(1.0)
        assert scaled == 1
        assert prefix == ''

        scaled, prefix = value_to_prefix(999.9)
        assert scaled == 999.9
        assert prefix == ''

    def test_zero_value(self):
        """Test that zero returns zero with no prefix."""
        scaled, prefix = value_to_prefix(0)
        assert scaled == 0.0
        assert prefix == ''

    def test_negative_values(self):
        """Test that negative values work correctly."""
        scaled, prefix = value_to_prefix(-1500000)
        assert scaled == -1.5
        assert prefix == 'M'

        scaled, prefix = value_to_prefix(-0.001)
        assert scaled == -1
        assert prefix == 'm'

    def test_precision_parameter(self):
        """Test the precision parameter."""
        # Auto precision (default)
        scaled, prefix = value_to_prefix(100.0, precision='auto')
        assert scaled == 100

        # Fixed precision
        scaled, prefix = value_to_prefix(100.0, precision=2)
        assert scaled == 100.0

        # Rounding with precision
        scaled, prefix = value_to_prefix(1234567, precision=2)
        assert scaled == 1.23
        assert prefix == 'M'

        # Full precision (no rounding)
        scaled, prefix = value_to_prefix(123.456789, precision=None)
        assert scaled == 123.456789  # Not rounded
        assert prefix == ''

        scaled, prefix = value_to_prefix(1234.56789, precision=None)
        assert math.isclose(scaled, 1.23456789)  # Not rounded, just scaled
        assert prefix == 'k'

    def test_measurement_precision_string(self):
        """Test measurement precision with string format."""
        # Round 0.1417 to nearest 1 milli
        scaled, prefix = value_to_prefix(0.1417, measurement_precision='1 m')
        assert scaled == 142
        assert prefix == 'm'

    def test_measurement_precision_float(self):
        """Test measurement precision with float value."""
        scaled, prefix = value_to_prefix(0.1417, measurement_precision=0.001)
        assert scaled == 142
        assert prefix == 'm'

    def test_exponential_fallback_true(self):
        """Test use_exp_fallback=True for values outside prefix range."""
        # Value too large (> 1e24)
        scaled, prefix = value_to_prefix(1e27, use_exp_fallback=True)
        assert scaled == 1e27
        assert prefix == ''

        # Value too small (< 1e-24)
        scaled, prefix = value_to_prefix(1e-27, use_exp_fallback=True)
        assert scaled == 1e-27
        assert prefix == ''

    def test_exponential_fallback_false(self):
        """Test use_exp_fallback=False clamps to nearest prefix."""
        # Value too large should clamp to Yotta (1e24)
        scaled, prefix = value_to_prefix(1e27, use_exp_fallback=False)
        assert scaled == 1000
        assert prefix == 'Y'

        # Value too small should clamp to yocto (1e-24)
        scaled, prefix = value_to_prefix(1e-27, use_exp_fallback=False)
        assert scaled == 0.001
        assert prefix == 'y'


class TestFormatWithPrefix:
    """Tests for format_with_prefix function."""

    def test_basic_formatting(self):
        """Test basic formatting with prefixes."""
        assert format_with_prefix(1500000) == '1.5 M'
        assert format_with_prefix(100.0) == '100'
        assert format_with_prefix(0.001) == '1 m'

    def test_spacing_parameter(self):
        """Test use_space parameter."""
        assert format_with_prefix(1500000, use_space=True) == '1.5 M'
        assert format_with_prefix(1500000, use_space=False) == '1.5M'

        # No prefix case should be unaffected
        assert format_with_prefix(100, use_space=True) == '100'
        assert format_with_prefix(100, use_space=False) == '100'

    def test_precision_formatting(self):
        """Test precision parameter in formatting."""
        assert format_with_prefix(100.0, precision=2) == '100.00'
        assert format_with_prefix(100.5, precision=0) == '100'
        assert format_with_prefix(1234567, precision=2) == '1.23 M'

    def test_measurement_precision_formatting(self):
        """Test formatting with measurement precision."""
        result = format_with_prefix(0.1417, measurement_precision='1 m')
        assert result == '142 m'

        result = format_with_prefix(0.1417, measurement_precision='1 m', use_space=False)
        assert result == '142m'

    def test_exponential_notation(self):
        """Test exponential notation for values outside prefix range."""
        # Very large value
        result = format_with_prefix(1e27)
        assert 'e+27' in result or 'e+27' in result.lower()

        # Very small value
        result = format_with_prefix(1e-27)
        assert 'e-27' in result or 'e-27' in result.lower()

    def test_auto_precision(self):
        """Test auto precision detection in formatting."""
        # Integer-like values
        assert format_with_prefix(1000.0, precision='auto') == '1 k'

        # Values with decimals
        assert format_with_prefix(1500.0, precision='auto') == '1.5 k'

    def test_full_precision(self):
        """Test full precision (no rounding) with precision=None."""
        # Value with many decimal places - shows full precision
        result = format_with_prefix(123.456789, precision=None)
        assert result == '123.456789'  # Full precision (up to 15 sig figs)

        # With prefix - clean output without FP artifacts
        result = format_with_prefix(1234.56789, precision=None)
        assert result == '1.23456789 k'

        # Compare with auto precision (capped at 3 decimals)
        result_auto = format_with_prefix(123.456789, precision='auto')
        assert result_auto == '123.457'  # Auto truncates

        # Full precision shows more detail than auto
        result_none = format_with_prefix(123.456789, precision=None)
        assert len(result_none) > len(result_auto)

        # Removes trailing zeros
        result = format_with_prefix(100.0, precision=None)
        assert result == '100'


class TestParsePrefixedValue:
    """Tests for parse_prefixed_value function."""

    def test_with_spaces(self):
        """Test parsing values with spaces between number and prefix."""
        assert parse_prefixed_value("100 m") == 0.1
        assert parse_prefixed_value("1.5 M") == 1500000.0
        assert parse_prefixed_value("1.5e3 k") == 1500000.0

    def test_without_spaces(self):
        """Test parsing values without spaces."""
        assert parse_prefixed_value("100m") == 0.1
        assert parse_prefixed_value("1.5M") == 1500000.0

    def test_no_prefix(self):
        """Test parsing values without prefixes."""
        assert parse_prefixed_value("100") == 100.0
        assert parse_prefixed_value("1.5") == 1.5

    def test_negative_values(self):
        """Test parsing negative values."""
        assert math.isclose(parse_prefixed_value("-50 u"), -5e-05)
        assert math.isclose(parse_prefixed_value("-100 m"), -0.1)

    def test_scientific_notation(self):
        """Test parsing values in scientific notation."""
        assert parse_prefixed_value("1e3") == 1000.0
        assert parse_prefixed_value("1e3 k") == 1e6
        assert parse_prefixed_value("1.5e-3 M") == 1500.0

    def test_invalid_inputs(self):
        """Test that invalid inputs raise ValueError."""
        # Empty string
        with pytest.raises(ValueError, match="Empty string"):
            parse_prefixed_value("")

        # Invalid prefix
        with pytest.raises(ValueError, match="Unknown prefix"):
            parse_prefixed_value("100 X")

        # Invalid number
        with pytest.raises(ValueError, match="Cannot parse numeric value"):
            parse_prefixed_value("abc m")

    def test_whitespace_handling(self):
        """Test that various whitespace formats are handled."""
        assert parse_prefixed_value("  100  m  ") == 0.1
        assert parse_prefixed_value("100   m") == 0.1


class TestDetectMinPrecision:
    """Tests for detect_min_precision helper function."""

    def test_integer_values(self):
        """Test precision detection for integer-like values."""
        assert detect_min_precision(100.0) == 0
        assert detect_min_precision(1.0) == 0

    def test_decimal_values(self):
        """Test precision detection for decimal values."""
        assert detect_min_precision(100.5) == 1
        assert detect_min_precision(100.12) == 2
        assert detect_min_precision(100.123) == 3

    def test_max_decimals_limit(self):
        """Test that max_decimals parameter limits precision."""
        # Value needs 4 decimals but max is 3
        result = detect_min_precision(100.1234, max_decimals=3)
        assert result == 3

        # Value needs 2 decimals and max is 3
        result = detect_min_precision(100.12, max_decimals=3)
        assert result == 2


class TestRoundToMeasurementPrecision:
    """Tests for _round_to_measurement_precision helper function."""

    def test_string_precision(self):
        """Test rounding with string precision format."""
        result = _round_to_measurement_precision(0.1417, '1 m')
        assert math.isclose(result, 0.142)

    def test_float_precision(self):
        """Test rounding with float precision value."""
        result = _round_to_measurement_precision(0.1417, 0.001)
        assert math.isclose(result, 0.142)

        result = _round_to_measurement_precision(123.456, 0.1)
        assert math.isclose(result, 123.5)

        result = _round_to_measurement_precision(1417, 10)
        assert math.isclose(result, 1420.0)

    def test_invalid_precision(self):
        """Test that non-positive precision raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            _round_to_measurement_precision(100, 0)

        with pytest.raises(ValueError, match="must be positive"):
            _round_to_measurement_precision(100, -1)


class TestDataDictionaries:
    """Tests for the SI prefix data structures."""

    def test_symbol_to_multiplier_completeness(self):
        """Test that SYMBOL_TO_MULTIPLIER contains all expected prefixes."""
        expected_symbols = ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k',
                           'm', 'u', 'n', 'p', 'f', 'a', 'z', 'y']
        assert set(SYMBOL_TO_MULTIPLIER.keys()) == set(expected_symbols)

    def test_multiplier_to_symbol_completeness(self):
        """Test that MULTIPLIER_TO_SYMBOL contains all expected multipliers."""
        expected_multipliers = [1e24, 1e21, 1e18, 1e15, 1e12, 1e9, 1e6, 1e3,
                               1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
        assert set(MULTIPLIER_TO_SYMBOL.keys()) == set(expected_multipliers)

    def test_bidirectional_mapping(self):
        """Test that symbol<->multiplier mappings are consistent."""
        for symbol, multiplier in SYMBOL_TO_MULTIPLIER.items():
            assert MULTIPLIER_TO_SYMBOL[multiplier] == symbol

        for multiplier, symbol in MULTIPLIER_TO_SYMBOL.items():
            assert SYMBOL_TO_MULTIPLIER[symbol] == multiplier


class TestRoundTripConversions:
    """Tests for round-trip conversions (format -> parse -> format)."""

    def test_format_parse_roundtrip(self):
        """Test that formatting and parsing are inverses."""
        test_values = [1500000, 0.001, 100, 1.5e6, 1.5e-6]

        for value in test_values:
            formatted = format_with_prefix(value, precision=6)
            parsed = parse_prefixed_value(formatted)
            assert math.isclose(parsed, value, rel_tol=1e-9)

    def test_prefix_multiplier_roundtrip(self):
        """Test that prefix<->multiplier conversions are inverses."""
        prefixes = ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k',
                   'm', 'u', 'n', 'p', 'f', 'a', 'z', 'y']

        for prefix in prefixes:
            multiplier = prefix_to_multiplier(prefix)
            recovered_prefix = multiplier_to_prefix(multiplier)
            assert recovered_prefix == prefix


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_non_zero_values(self):
        """Test handling of very small but non-zero values."""
        scaled, prefix = value_to_prefix(1e-30)
        assert prefix == ''  # Outside range with fallback

        scaled, prefix = value_to_prefix(1e-23)
        assert prefix == 'y'  # Within range

    def test_very_large_values(self):
        """Test handling of very large values."""
        scaled, prefix = value_to_prefix(1e30)
        assert prefix == ''  # Outside range with fallback

        # 1e23 is closer to 1e21 (Z) than 1e24 (Y) on log scale
        scaled, prefix = value_to_prefix(1e23)
        assert prefix == 'Z'  # Within range

        # Test actual Yotta range
        scaled, prefix = value_to_prefix(5e24)
        assert prefix == 'Y'

    def test_boundary_values(self):
        """Test values at boundaries between prefixes."""
        # At 1000 boundary
        scaled, prefix = value_to_prefix(999.99)
        assert prefix == ''

        scaled, prefix = value_to_prefix(1000.0)
        assert prefix == 'k'

        # At 1 boundary
        scaled, prefix = value_to_prefix(1.0)
        assert prefix == ''

        scaled, prefix = value_to_prefix(0.999)
        assert prefix == 'm'

    def test_precision_edge_cases(self):
        """Test edge cases in precision handling."""
        # Precision of 0 should show no decimals
        result = format_with_prefix(1234.5678, precision=0)
        assert '.' not in result.split()[0]  # No decimal in number part

        # Very high precision
        result = format_with_prefix(1.23456789, precision=8)
        assert '1.23456789' in result
