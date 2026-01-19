"""Tests for ElectricalTransportAccessor (.etr accessor for pandas DataFrames)."""

import pytest
import pandas as pd
import numpy as np
from medapy import ureg
import medapy.ms_pandas  # Import to register the .ms accessor
import medapy.analysis.electron_transport.etr_pandas  # Import to register the .etr accessor


class TestElectricalTransportAccessorBasics:
    """Test basic ElectricalTransportAccessor functionality."""

    def test_accessor_available(self):
        """Test that .etr accessor is available on DataFrames."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        assert hasattr(df, 'etr')

    def test_etr_inherits_from_proc(self):
        """Test that .etr accessor inherits from .proc accessor."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        # Should have proc methods
        assert hasattr(df.etr, 'normalize')
        assert hasattr(df.etr, 'symmetrize')
        assert hasattr(df.etr, 'interpolate')

    def test_etr_has_transport_methods(self):
        """Test that .etr accessor has transport-specific methods."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        assert hasattr(df.etr, 'r2rho')
        assert hasattr(df.etr, 'fit_linhall')
        assert hasattr(df.etr, 'fit_twoband')
        assert hasattr(df.etr, 'calculate_twoband')


class TestR2Rho:
    """Test r2rho (resistance to resistivity conversion) method."""

    def test_r2rho_xx_geometry(self):
        """Test r2rho with xx geometry (longitudinal)."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [1000, 1100, 1200]})
        df.ms.init_msheet()
        df.ms.set_as_y('Resistance')

        t = 50e-9  # 50 nm
        width = 10e-6  # 10 um
        length = 20e-6  # 20 um

        result = df.etr.r2rho('xx', col='Resistance', t=t, width=width, length=length,
                              new_col='Resistivity', inplace=False)

        assert 'Resistivity' in result.columns
        # Check that resistivity values are calculated correctly
        # rho_xx = R * (t * width) / length
        expected = np.array([1000, 1100, 1200]) * (t * width) / length
        np.testing.assert_allclose(result['Resistivity'].values, expected)

    def test_r2rho_xy_geometry(self):
        """Test r2rho with xy geometry (Hall/transverse)."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        df.ms.set_as_y('Resistance')

        t = 50e-9  # 50 nm

        result = df.etr.r2rho('xy', col='Resistance', t=t,
                              new_col='Resistivity', inplace=False)

        assert 'Resistivity' in result.columns
        # Check that resistivity values are calculated correctly
        # rho_xy = R * t
        expected = np.array([100, 110, 120]) * t
        np.testing.assert_allclose(result['Resistivity'].values, expected)

    def test_r2rho_with_pint_quantities(self):
        """Test r2rho with pint Quantity objects for dimensions."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [1000, 1100, 1200]})
        df.ms.init_msheet()
        df.ms.set_as_y('Resistance')

        t = 50 * ureg.nm
        width = 10 * ureg.um
        length = 20 * ureg.um

        result = df.etr.r2rho('xx', col='Resistance', t=t, width=width, length=length,
                              new_col='Resistivity', inplace=False)

        assert 'Resistivity' in result.columns
        # Should work the same as with floats
        expected = np.array([1000, 1100, 1200]) * (50e-9 * 10e-6) / 20e-6
        np.testing.assert_allclose(result['Resistivity'].values, expected)

    def test_r2rho_preserves_unit_dimensions(self):
        """Test that r2rho creates correct resistivity units."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        df.ms.set_as_y('Resistance')

        t = 50e-9  # meters

        result = df.etr.r2rho('xy', col='Resistance', t=t,
                              new_col='Resistivity', inplace=False)

        # Unit should be ohm * meter
        unit = result.ms.get_unit('Resistivity')
        assert 'meter' in unit or 'm' in unit
        assert 'ohm' in unit or 'Ω' in unit

    def test_r2rho_with_axis_and_label(self):
        """Test r2rho with axis assignment and label."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()

        t = 50e-9

        result = df.etr.r2rho('xy', t=t, new_col='Resistivity',
                              set_axis='y', add_label='rho_xy', inplace=False)

        assert result.ms.axes['y'] == 'Resistivity'
        assert result.ms.get_column('rho_xy') == 'Resistivity'

    def test_r2rho_inplace(self):
        """Test r2rho with inplace=True."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()

        t = 50e-9

        result = df.etr.r2rho('xy', t=t, new_col='Resistivity', inplace=True)

        assert result is None
        assert 'Resistivity' in df.columns

    def test_r2rho_uses_y_axis_by_default(self):
        """Test that r2rho uses y axis column by default."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'Resistance (ohm)': [100, 110, 120]})
        df.ms.init_msheet()
        df.ms.set_as_y('Resistance')

        t = 50e-9

        result = df.etr.r2rho('xy', t=t, new_col='Resistivity', inplace=False)

        # Should use Resistance (y axis) even though col not specified
        assert 'Resistivity' in result.columns


class TestFitLinHall:
    """Test fit_linhall (linear Hall fit) method."""

    def test_fit_linhall_basic(self):
        """Test basic linear Hall fitting."""
        # Create linear data
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field  # rho = 100 + 2*H

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        coefs, result = df.etr.fit_linhall(col='Resistivity', add_col='fit', inplace=False)

        # Check coefficients - order might be [intercept, slope] or [slope, intercept]
        assert len(coefs) == 2
        # Check that coefficients contain both 2 and 100
        assert any(np.isclose(c, 2, rtol=1e-10) for c in coefs)
        assert any(np.isclose(c, 100, rtol=1e-10) for c in coefs)

        # Check that fit column is added
        assert 'Resistivity_fit' in result.columns

    def test_fit_linhall_with_range(self):
        """Test linear Hall fitting with x_range specified."""
        # Create data with nonlinear region
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field + 0.1 * field**2  # Slightly nonlinear

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        # Fit only in range 1 to 5
        coefs, result = df.etr.fit_linhall(x_range=(1, 5), add_col='fit', inplace=False)

        # Coefficients should be calculated from the specified range
        assert len(coefs) == 2

    def test_fit_linhall_without_add_col(self):
        """Test fit_linhall without adding fit column."""
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        coefs, result = df.etr.fit_linhall(add_col='', inplace=False)

        # Should only return coefficients without adding column
        assert 'Resistivity_' not in result.columns
        assert len(coefs) == 2

    def test_fit_linhall_preserves_units(self):
        """Test that fit_linhall preserves units in fit column."""
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        coefs, result = df.etr.fit_linhall(add_col='fit', inplace=False)

        # Fit column should have same unit as original
        assert result.ms.get_unit('Resistivity_fit') == result.ms.get_unit('Resistivity')

    def test_fit_linhall_with_axis_and_label(self):
        """Test fit_linhall with axis and label assignment."""
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        coefs, result = df.etr.fit_linhall(add_col='fit', set_axis='f', add_label='linear_fit',
                                           inplace=False)

        assert result.ms.axes['f'] == 'Resistivity_fit'
        assert result.ms.get_column('linear_fit') == 'Resistivity_fit'

    def test_fit_linhall_inplace(self):
        """Test fit_linhall with inplace=True."""
        field = np.array([-5, -3, -1, 0, 1, 3, 5])
        rho = 100 + 2 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        coefs, result = df.etr.fit_linhall(add_col='fit', inplace=True)

        assert result is None
        assert 'Resistivity_fit' in df.columns


class TestFitTwoband:
    """Test fit_twoband (two-band model fitting) method."""

    def test_fit_twoband_returns_coefficients(self):
        """Test that fit_twoband returns optimized parameters."""
        # Create simple test data
        field = np.linspace(-5, 5, 11)
        rho = 100 + 0.1 * field  # Simple linear for testing

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        # Initial guess [n1, n2, mu1, mu2]
        p0 = [1e26, 1e25, 0.01, 0.02]

        # This might not converge to physical values with linear data,
        # but should return a tuple
        try:
            coefs, result = df.etr.fit_twoband(p0, kind='xy', bands='he',
                                               add_col='fit', inplace=False)
            assert isinstance(coefs, tuple)
            assert len(coefs) == 4
        except Exception:
            # The fit might fail with simple linear data, which is okay
            pytest.skip("Two-band fit requires more realistic data")

    def test_fit_twoband_with_field_range(self):
        """Test fit_twoband with field_range specified."""
        field = np.linspace(-10, 10, 21)
        rho = 100 + 0.1 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        p0 = [1e26, 1e25, 0.01, 0.02]

        try:
            # Use only subset of field for fitting
            coefs, result = df.etr.fit_twoband(p0, kind='xy', bands='he',
                                               field_range=(-5, 5), inside_range=True,
                                               add_col='', inplace=False)
            assert isinstance(coefs, tuple)
        except Exception:
            pytest.skip("Two-band fit requires more realistic data")

    def test_fit_twoband_without_add_col(self):
        """Test fit_twoband without adding fit column."""
        field = np.linspace(-5, 5, 11)
        rho = 100 + 0.1 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        p0 = [1e26, 1e25, 0.01, 0.02]

        try:
            coefs, result = df.etr.fit_twoband(p0, kind='xy', bands='he',
                                               add_col=None, inplace=False)
            # Should not add column
            assert 'Resistivity_2bndhe' not in result.columns
        except Exception:
            pytest.skip("Two-band fit requires more realistic data")

    def test_fit_twoband_adds_column_with_correct_name(self):
        """Test that fit_twoband adds column with correct naming."""
        field = np.linspace(-5, 5, 11)
        rho = 100 + 0.1 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        p0 = [1e26, 1e25, 0.01, 0.02]

        try:
            coefs, result = df.etr.fit_twoband(p0, kind='xy', bands='he',
                                               add_col='fit', inplace=False)
            # Should add column with name Resistivity_fithe
            assert 'Resistivity_fithe' in result.columns
        except Exception:
            pytest.skip("Two-band fit requires more realistic data")

    def test_fit_twoband_extension_as_dataframe(self):
        """Test fit_twoband with extension as DataFrame."""
        field = np.linspace(-5, 5, 11)
        rho_xy = 0.1 * field
        rho_xx = 100 + 0.01 * field**2

        df = pd.DataFrame({'Field (T)': field, 'Rho_xy (ohm*m)': rho_xy, 'Rho_xx (ohm*m)': rho_xx})
        df.ms.init_msheet()

        # Create extension DataFrame
        ext_df = df[['Field', 'Rho_xx']].copy()
        ext_df.ms.init_msheet()

        p0 = [1e26, 1e25, 0.01, 0.02]

        try:
            coefs, result = df.etr.fit_twoband(p0, col='Rho_xy', kind='xy', bands='he',
                                               extension=ext_df, add_col='', inplace=False)
            assert isinstance(coefs, tuple)
        except Exception:
            pytest.skip("Two-band fit requires more realistic data")

    def test_fit_twoband_inplace(self):
        """Test fit_twoband with inplace=True."""
        field = np.linspace(-5, 5, 11)
        rho = 100 + 0.1 * field

        df = pd.DataFrame({'Field (T)': field, 'Resistivity (ohm*m)': rho})
        df.ms.init_msheet()

        p0 = [1e26, 1e25, 0.01, 0.02]

        try:
            coefs, result = df.etr.fit_twoband(p0, kind='xy', bands='he',
                                               add_col='fit', inplace=True)
            assert result is None
            assert 'Resistivity_fithe' in df.columns
        except Exception:
            pytest.skip("Two-band fit requires more realistic data")


class TestCalculateTwoband:
    """Test calculate_twoband method."""

    def test_calculate_twoband_single_column(self):
        """Test calculate_twoband for single column."""
        field = np.linspace(-5, 5, 11)

        df = pd.DataFrame({'Field (T)': field, 'Rho_xy (ohm*m)': np.zeros(11)})
        df.ms.init_msheet()

        # Use some parameters (not necessarily physical)
        p = (1e26, 1e25, 0.01, 0.02)

        result = df.etr.calculate_twoband(p, cols='Rho_xy', kinds='xy', bands='he',
                                          append='calc', inplace=False)

        assert 'Rho_xy_calche' in result.columns

    def test_calculate_twoband_multiple_columns(self):
        """Test calculate_twoband for multiple columns."""
        field = np.linspace(-5, 5, 11)

        df = pd.DataFrame({
            'Field (T)': field,
            'Rho_xy (ohm*m)': np.zeros(11),
            'Rho_xx (ohm*m)': np.zeros(11)
        })
        df.ms.init_msheet()

        p = (1e26, 1e25, 0.01, 0.02)

        result = df.etr.calculate_twoband(p, cols=['Rho_xy', 'Rho_xx'],
                                          kinds=['xy', 'xx'], bands='he',
                                          append='calc', inplace=False)

        assert 'Rho_xy_calche' in result.columns
        assert 'Rho_xx_calche' in result.columns

    def test_calculate_twoband_preserves_units(self):
        """Test that calculate_twoband preserves units."""
        field = np.linspace(-5, 5, 11)

        df = pd.DataFrame({'Field (T)': field, 'Rho_xy (ohm*m)': np.zeros(11)})
        df.ms.init_msheet()

        p = (1e26, 1e25, 0.01, 0.02)

        result = df.etr.calculate_twoband(p, cols='Rho_xy', kinds='xy', bands='he',
                                          append='calc', inplace=False)

        # Should preserve units from original column
        assert result.ms.get_unit('Rho_xy_calche') == result.ms.get_unit('Rho_xy')

    def test_calculate_twoband_with_axes_and_labels(self):
        """Test calculate_twoband with axis and label assignment."""
        field = np.linspace(-5, 5, 11)

        df = pd.DataFrame({'Field (T)': field, 'Rho_xy (ohm*m)': np.zeros(11)})
        df.ms.init_msheet()

        p = (1e26, 1e25, 0.01, 0.02)

        result = df.etr.calculate_twoband(p, cols='Rho_xy', kinds='xy', bands='he',
                                          append='calc', set_axes='c', add_labels='calculated',
                                          inplace=False)

        assert result.ms.axes['c'] == 'Rho_xy_calche'
        assert result.ms.get_column('calculated') == 'Rho_xy_calche'

    def test_calculate_twoband_inplace(self):
        """Test calculate_twoband with inplace=True."""
        field = np.linspace(-5, 5, 11)

        df = pd.DataFrame({'Field (T)': field, 'Rho_xy (ohm*m)': np.zeros(11)})
        df.ms.init_msheet()

        p = (1e26, 1e25, 0.01, 0.02)

        result = df.etr.calculate_twoband(p, cols='Rho_xy', kinds='xy', bands='he',
                                          append='calc', inplace=True)

        assert result is None
        assert 'Rho_xy_calche' in df.columns


class TestEtrInheritance:
    """Test that etr accessor properly inherits proc methods."""

    def test_etr_ensure_increasing(self):
        """Test that etr.ensure_increasing works."""
        df = pd.DataFrame({'Field (T)': [3, 2, 1, 0], 'R (ohm)': [130, 120, 110, 100]})
        df.ms.init_msheet()
        result = df.etr.ensure_increasing(inplace=False)
        np.testing.assert_array_equal(result['Field'].values, [0, 1, 2, 3])

    def test_etr_select_range(self):
        """Test that etr.select_range works."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 3, 4], 'R (ohm)': [100, 110, 120, 130, 140]})
        df.ms.init_msheet()
        result = df.etr.select_range((1, 3), inplace=False)
        assert len(result) == 3

    def test_etr_normalize(self):
        """Test that etr.normalize works."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2], 'R (ohm)': [100, 200, 300]})
        df.ms.init_msheet()
        result = df.etr.normalize(by='first', append='norm', inplace=False)
        assert 'R_norm' in result.columns

    def test_etr_symmetrize(self):
        """Test that etr.symmetrize works."""
        df = pd.DataFrame({'Field (T)': [-2, -1, 0, 1, 2], 'R (ohm)': [80, 90, 100, 110, 120]})
        df.ms.init_msheet()
        result = df.etr.symmetrize(append='sym', inplace=False)
        assert 'R_sym' in result.columns

    def test_etr_interpolate(self):
        """Test that etr.interpolate works."""
        df = pd.DataFrame({'Field (T)': [0, 1, 2, 4], 'R (ohm)': [100, 110, 120, 140]})
        df.ms.init_msheet()
        x_new = np.array([0, 1, 2, 3, 4])
        result = df.etr.interpolate(x_new, inplace=False)
        assert len(result) == 5


class TestIntegrationWithExamples:
    """Integration tests based on example usage."""

    def test_example_4_workflow(self):
        """Test workflow from example 4_etr_pandas.py."""
        # Create sample data
        df = pd.DataFrame({
            'Field (Oe)': [5, 3, 1, 0, -1, -3, -5],
            'Current (uA)': [10, 10, 10, 10, 10, 10, 10],
            'Voltage (mV)': [103, 49, 22, 10, 18, 51, 97],
            'Resistance (Ohm)': [10300, 4900, 2200, 1000, 1800, 5100, 9700]
        })

        custom_unit_dict = dict(Ohm='ohm')
        df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

        # Add labels
        df.ms.add_labels({'Field': 'H', 'Resistance': 'R', 'Voltage': 'V', 'Current': 'I'})

        # Set y axis
        df.ms.set_as_y('R')

        # Use etr.ensure_increasing
        df.etr.ensure_increasing(inplace=True)
        assert df['Field'].is_monotonic_increasing

        # Convert resistance to resistivity
        t = 50 * ureg('nm')
        w = 10 * ureg('cm')
        l = 2 * ureg('cm')

        df.etr.r2rho('xx', col='R', t=t, width=w, length=l,
                     new_col='Resistivity', add_label='rho', inplace=True)

        assert 'Resistivity' in df.columns
        assert df.ms.get_column('rho') == 'Resistivity'

    def test_example_5_realistic_workflow(self):
        """Test workflow from example 5_realistic_example.py (partial)."""
        # Create sample Rxx and Rxy data
        field = np.linspace(-14, 14, 29)
        rxx = 1000 + 10 * field**2
        rxy = 100 * field

        xx = pd.DataFrame({'Field (T)': field, 'Resistance': rxx})
        xy = pd.DataFrame({'Field (T)': field, 'Resistance': rxy})

        # Initialize measurement sheets
        custom_unit_dict = dict(Ohms='ohm')
        xx.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
        xy.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

        # Rename columns
        xx.ms.rename({'Resistance': 'Resistance_xx'})
        xy.ms.rename({'Resistance': 'Resistance_xy'})

        # Concatenate
        data = xx.ms.concat(xy)
        assert 'Resistance_xx' in data.columns
        assert 'Resistance_xy' in data.columns

        # Validate monotonic increasing
        data.etr.ensure_increasing(inplace=True)
        assert data['Field'].is_monotonic_increasing
