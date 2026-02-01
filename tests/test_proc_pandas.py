"""Tests for DataProcessingAccessor (.proc accessor for pandas DataFrames)."""

import pytest
import pandas as pd
import numpy as np
from medapy import ureg
import medapy.ms_pandas  # Import to register the .ms accessor
import medapy.analysis.proc_pandas  # Import to register the .proc accessor


class TestDataProcessingAccessorBasics:
    """Test basic DataProcessingAccessor functionality."""

    def test_accessor_available(self):
        """Test that .proc accessor is available on DataFrames."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        assert hasattr(df, 'proc')

    def test_accessor_requires_initialized_msheet(self):
        """Test that .proc accessor requires initialized MSheet."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with pytest.raises(AttributeError, match="MeasurementSheet must be initialized"):
            _ = df.proc

    def test_accessor_requires_two_columns(self):
        """Test that .proc accessor requires at least 2 columns."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.ms.init_msheet()
        with pytest.raises(ValueError, match="at least two columns"):
            _ = df.proc

    def test_ms_property(self):
        """Test that .proc.ms property returns MeasurementSheetAccessor."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        assert df.proc.ms is df.ms

    def test_axis_properties(self):
        """Test that .proc has x, y, z properties."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        np.testing.assert_array_equal(df.proc.x.values, [0, 1, 2])
        np.testing.assert_array_equal(df.proc.y.values, [100, 110, 120])


class TestCheckMonotonic:
    """Test check_monotonic method."""

    def test_check_monotonic_increasing(self):
        """Test check_monotonic with increasing data."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3], 'R (ohm)': [100, 110, 120, 130]})
        df.ms.init_msheet()
        # check_monotonic expects monotonic data
        result = df.proc.check_monotonic()
        assert result != 0  # Should be monotonic (either 1 or -1)

    def test_check_monotonic_decreasing(self):
        """Test check_monotonic with decreasing data."""
        df = pd.DataFrame({'Field (T)': [3, 2, 1, 0], 'R (ohm)': [130, 120, 110, 100]})
        df.ms.init_msheet()
        result = df.proc.check_monotonic()
        assert result != 0  # Should be monotonic (either 1 or -1)

    def test_check_monotonic_non_monotonic(self):
        """Test check_monotonic with non-monotonic data."""
        df = pd.DataFrame({'Field (T)': [0, 2, 1, 3], 'R (ohm)': [100, 120, 110, 130]})
        df.ms.init_msheet()
        result = df.proc.check_monotonic()
        assert result == 0  # Should not be monotonic

    def test_check_monotonic_interrupt_raises(self):
        """Test check_monotonic with interrupt=True raises on non-monotonic data."""
        df = pd.DataFrame({'Field (T)': [0, 2, 1, 3], 'R (ohm)': [100, 120, 110, 130]})
        df.ms.init_msheet()
        with pytest.raises(ValueError, match="not monotonic"):
            df.proc.check_monotonic(interrupt=True)


class TestEnsureIncreasing:
    """Test ensure_increasing method."""

    def test_ensure_increasing_already_increasing(self):
        """Test ensure_increasing with already increasing data."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3], 'R (ohm)': [100, 110, 120, 130]})
        df.ms.init_msheet()
        result = df.proc.ensure_increasing(inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [0, 1, 2, 3])
        np.testing.assert_array_equal(result['R'].values, [100, 110, 120, 130])

    def test_ensure_increasing_decreasing_data(self):
        """Test ensure_increasing reverses decreasing data."""
        df = pd.DataFrame({'Field (T)': [3, 2, 1, 0], 'R (ohm)': [130, 120, 110, 100]})
        df.ms.init_msheet()
        result = df.proc.ensure_increasing(inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [0, 1, 2, 3])
        np.testing.assert_array_equal(result['R'].values, [100, 110, 120, 130])

    def test_ensure_increasing_non_monotonic_raises(self):
        """Test ensure_increasing raises on non-monotonic data."""
        df = pd.DataFrame({'Field (T)': [0, 2, 1, 3], 'R (ohm)': [100, 120, 110, 130]})
        df.ms.init_msheet()
        with pytest.raises(ValueError, match="not monotonic"):
            df.proc.ensure_increasing(inplace=False)

    def test_ensure_increasing_inplace(self):
        """Test ensure_increasing with inplace=True."""
        df = pd.DataFrame({'Field (T)': [3, 2, 1, 0], 'R (ohm)': [130, 120, 110, 100]})
        df.ms.init_msheet()
        result = df.proc.ensure_increasing(inplace=True)
        assert result is None
        np.testing.assert_array_equal(df['Field'].values, [0, 1, 2, 3])
        np.testing.assert_array_equal(df['R'].values, [100, 110, 120, 130])

    def test_ensure_increasing_preserves_metadata(self):
        """Test that ensure_increasing preserves MSheet metadata."""
        df = pd.DataFrame({'Field (T)': [3, 2, 1, 0], 'R (ohm)': [130, 120, 110, 100]})
        df.ms.init_msheet()
        df.ms.add_label('Field', 'H')
        result = df.proc.ensure_increasing(inplace=False)
        assert result.ms.get_column('H') == 'Field'
        assert result.ms.get_unit('Field') == 'T'


class TestSelectRange:
    """Test select_range method."""

    def test_select_range_inside(self):
        """Test selecting data inside a range."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4, 5], 'R (ohm)': [100, 110, 120, 130, 140, 150]})
        df.ms.init_msheet()
        result = df.proc.select_range((1, 4), inside_range=True, inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [1, 2, 3, 4])
        assert len(result) == 4

    def test_select_range_outside(self):
        """Test selecting data outside a range."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4, 5], 'R (ohm)': [100, 110, 120, 130, 140, 150]})
        df.ms.init_msheet()
        result = df.proc.select_range((2, 4), inside_range=False, inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [0, 1, 5])

    def test_select_range_inclusive_both(self):
        """Test select_range with inclusive='both'."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.select_range((1, 3), inclusive='both', inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [1, 2, 3])

    def test_select_range_inclusive_neither(self):
        """Test select_range with inclusive='neither'."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.select_range((1, 3), inclusive='neither', inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [2])

    def test_select_range_inplace(self):
        """Test select_range with inplace=True."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.select_range((1, 3), inplace=True)
        assert result is None
        np.testing.assert_array_equal(df['Field'].values, [1, 2, 3])

    def test_select_range_resets_index(self):
        """Test that select_range resets the index."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.select_range((2, 4), inplace=False)
        np.testing.assert_array_equal(result.index.values, [0, 1, 2])


class TestNormalize:
    """Test normalize method."""

    def test_normalize_by_first(self):
        """Test normalizing by first value."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', inplace=False)
        np.testing.assert_array_equal(result['R_norm'].values, [1.0, 2.0, 3.0])

    def test_normalize_by_mid(self):
        """Test normalizing by middle value."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='mid', append='norm', inplace=False)
        np.testing.assert_array_equal(result['R_norm'].values, [0.5, 1.0, 1.5])

    def test_normalize_by_last(self):
        """Test normalizing by last value."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='last', append='norm', inplace=False)
        np.testing.assert_allclose(result['R_norm'].values, [1/3, 2/3, 1.0])

    def test_normalize_by_value(self):
        """Test normalizing by specific value."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by=50, append='norm', inplace=False)
        np.testing.assert_array_equal(result['R_norm'].values, [2.0, 4.0, 6.0])

    def test_normalize_overwrites_column(self):
        """Test normalizing overwrites column when append is empty."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='', inplace=False)
        np.testing.assert_array_equal(result['R'].values, [1.0, 2.0, 3.0])
        # Should have dimensionless unit (which is '1' in medapy)
        assert result.ms.get_unit('R') == '1'

    def test_normalize_sets_dimensionless(self):
        """Test that normalize sets unit to dimensionless."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', inplace=False)
        # Dimensionless unit is '1' in medapy
        assert result.ms.get_unit('R_norm') == '1'

    def test_normalize_with_axis_and_label(self):
        """Test normalize with set_axes and add_labels."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', set_axes='n', add_labels='normalized', inplace=False)
        assert result.ms.axes['n'] == 'R_norm'
        assert result.ms.get_column('normalized') == 'R_norm'

    def test_normalize_multiple_columns(self):
        """Test normalizing multiple columns."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300], 'V (V)': [10, 20, 30]})
        df.ms.init_msheet()
        result = df.proc.normalize(cols=['R', 'V'], by='first', append='norm', inplace=False)
        assert 'R_norm' in result.columns
        assert 'V_norm' in result.columns
        np.testing.assert_array_equal(result['R_norm'].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result['V_norm'].values, [1.0, 2.0, 3.0])


class TestSymmetrize:
    """Test symmetrize method."""

    def test_symmetrize_basic(self):
        """Test basic symmetrization."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.symmetrize(append='sym', inplace=False)
        # Symmetrized value at i should be (y[i] + y[-(i+1)]) / 2
        expected = np.array([100, 100, 100, 100, 100])
        np.testing.assert_array_equal(result['R_sym'].values, expected)

    def test_symmetrize_overwrites_column(self):
        """Test symmetrize overwrites when append is empty."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.symmetrize(append='', inplace=False)
        assert 'R_sym' not in result.columns
        assert 'R' in result.columns

    def test_symmetrize_preserves_units(self):
        """Test that symmetrize preserves units."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.symmetrize(append='sym', inplace=False)
        assert result.ms.get_unit('R_sym') == result.ms.get_unit('R')

    def test_symmetrize_multiple_columns(self):
        """Test symmetrizing multiple columns."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120], 'V (V)': [8, 9, 10, 11, 12]})
        df.ms.init_msheet()
        result = df.proc.symmetrize(cols=['R', 'V'], append='sym', inplace=False)
        assert 'R_sym' in result.columns
        assert 'V_sym' in result.columns


class TestAntisymmetrize:
    """Test antisymmetrize method."""

    def test_antisymmetrize_basic(self):
        """Test basic antisymmetrization."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.antisymmetrize(append='asym', inplace=False)
        # Antisymmetrized value at i should be (y[i] - y[-(i+1)]) / 2
        # For [80, 90, 100, 110, 120]:
        # i=0: (80 - 120)/2 = -20
        # i=1: (90 - 110)/2 = -10
        # i=2: (100 - 100)/2 = 0
        # i=3: (110 - 90)/2 = 10
        # i=4: (120 - 80)/2 = 20
        expected = np.array([-20, -10, 0, 10, 20])
        np.testing.assert_array_equal(result['R_asym'].values, expected)

    def test_antisymmetrize_preserves_units(self):
        """Test that antisymmetrize preserves units."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.antisymmetrize(append='asym', inplace=False)
        assert result.ms.get_unit('R_asym') == result.ms.get_unit('R')


class TestInterpolate:
    """Test interpolate method."""

    def test_interpolate_linear(self):
        """Test linear interpolation."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 4], 'R (ohm)': [100, 110, 120, 140]})
        df.ms.init_msheet()
        x_new = np.array([0, 0.5, 1, 1.5, 2, 3, 4])
        result = df.proc.interpolate(x_new, inplace=False)
        # Check that x is updated
        np.testing.assert_array_equal(result['Field'].values, x_new)
        # Check interpolated values
        expected = np.array([100, 105, 110, 115, 120, 130, 140])
        np.testing.assert_array_equal(result['R'].values, expected)

    def test_interpolate_preserves_units(self):
        """Test that interpolate preserves units."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 4], 'R (ohm)': [100, 110, 120, 140]})
        df.ms.init_msheet()
        x_new = np.array([0, 2, 4])
        result = df.proc.interpolate(x_new, inplace=False)
        assert result.ms.get_unit('Field') == 'T'
        assert result.ms.get_unit('R') == 'Ω'

    def test_interpolate_multiple_columns(self):
        """Test interpolating multiple columns."""
        df = pd.DataFrame({
            'Field (T)': [0, 1, 2],
            'R (ohm)': [100, 110, 120],
            'V (V)': [10, 15, 20]
        })
        df.ms.init_msheet()
        x_new = np.array([0, 0.5, 1, 1.5, 2])
        result = df.proc.interpolate(x_new, cols=['R', 'V'], inplace=False)
        assert len(result) == 5
        assert 'R' in result.columns
        assert 'V' in result.columns
        np.testing.assert_array_equal(result['R'].values, [100, 105, 110, 115, 120])
        np.testing.assert_array_equal(result['V'].values, [10, 12.5, 15, 17.5, 20])

    def test_interpolate_inplace(self):
        """Test interpolate with inplace=True."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        x_new = np.array([0, 1, 2])
        result = df.proc.interpolate(x_new, inplace=True)
        assert result is None
        np.testing.assert_array_equal(df['Field'].values, x_new)


class TestMovingAverage:
    """Test moving_average method."""

    def test_moving_average_basic(self):
        """Test basic moving average."""
        df = pd.DataFrame({'Field (T)': [0.0, 1.0, 2.0, 3.0, 4.0], 'R (ohm)': [100.0, 110.0, 120.0, 130.0, 140.0]})
        df.ms.init_msheet()
        result = df.proc.moving_average(window=3, append='ma', inplace=False)
        # With window=3, center point should be average of 3 points
        assert 'R_ma' in result.columns
        assert len(result) == 5

    def test_moving_average_preserves_units(self):
        """Test that moving_average preserves units."""
        df = pd.DataFrame({'Field (T)': [0.0, 1.0, 2.0, 3.0, 4.0], 'R (ohm)': [100.0, 110.0, 120.0, 130.0, 140.0]})
        df.ms.init_msheet()
        result = df.proc.moving_average(window=3, append='ma', inplace=False)
        assert result.ms.get_unit('R_ma') == result.ms.get_unit('R')

    def test_moving_average_edge_mode_drop(self):
        """Test moving_average with edge_mode='drop'."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.moving_average(window=3, edge_mode='drop', append='ma', inplace=False)
        # With window=3, drops 1 from each edge
        assert len(result) == 3


class TestSavgolFilter:
    """Test savgol_filter method."""

    def test_savgol_filter_basic(self):
        """Test basic Savitzky-Golay filter."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.savgol_filter(window=5, order=2, append='sg', inplace=False)
        assert 'R_sg' in result.columns
        assert len(result) == 5

    def test_savgol_filter_preserves_units(self):
        """Test that savgol_filter preserves units."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.proc.savgol_filter(window=5, order=2, append='sg', inplace=False)
        assert result.ms.get_unit('R_sg') == result.ms.get_unit('R')

    def test_savgol_filter_edge_mode_drop(self):
        """Test savgol_filter with edge_mode='drop'."""
        df = pd.DataFrame({
            'Field (T)': list(range(10)),
            'R (ohm)': [100 + i*10 for i in range(10)]
        })
        df.ms.init_msheet()
        result = df.proc.savgol_filter(window=5, order=2, edge_mode='drop', append='sg', inplace=False)
        # With window=5, drops 2 from each edge
        assert len(result) == 6


class TestInplaceParameter:
    """Test inplace parameter across all methods."""

    def test_inplace_true_returns_none(self):
        """Test that inplace=True returns None."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', inplace=True)
        assert result is None

    def test_inplace_true_modifies_original(self):
        """Test that inplace=True modifies original DataFrame."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        df.proc.normalize(by='first', append='norm', inplace=True)
        assert 'R_norm' in df.columns

    def test_inplace_false_returns_dataframe(self):
        """Test that inplace=False returns DataFrame."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', inplace=False)
        assert isinstance(result, pd.DataFrame)

    def test_inplace_false_preserves_original(self):
        """Test that inplace=False preserves original DataFrame."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        result = df.proc.normalize(by='first', append='norm', inplace=False)
        assert 'R_norm' not in df.columns
        assert 'R_norm' in result.columns
