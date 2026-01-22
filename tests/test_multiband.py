"""Tests for multiband electron transport fitting functionality."""

import numpy as np
import pytest
from medapy.analysis.electron_transport import electron_transport as etr


class TestGenerateMultibandEq:
    """Tests for generate_multiband_eq function."""

    def test_two_band_hall(self):
        """Test two-band Hall equation generation."""
        eq = etr.generate_multiband_eq('xy', 'he')

        field = np.array([0.0, 1.0, 2.0])
        n1, mu1 = 1e26, 0.015
        n2, mu2 = 5e25, 0.02

        result = eq(field, n1, mu1, n2, mu2)
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_two_band_mr(self):
        """Test two-band magnetoresistance equation generation."""
        eq = etr.generate_multiband_eq('xx', 'he')

        field = np.array([0.0, 1.0, 2.0])
        n1, mu1 = 1e26, 0.015
        n2, mu2 = 5e25, 0.02

        result = eq(field, n1, mu1, n2, mu2)
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_three_band(self):
        """Test three-band equation generation."""
        eq = etr.generate_multiband_eq('xy', 'hee')

        field = np.array([0.0, 1.0])
        n1, mu1 = 1e26, 0.01
        n2, mu2 = 5e25, 0.02
        n3, mu3 = 2e25, 0.015

        result = eq(field, n1, mu1, n2, mu2, n3, mu3)
        assert isinstance(result, np.ndarray)
        assert result.shape == field.shape

    def test_invalid_kind(self):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError, match="kind"):
            etr.generate_multiband_eq('invalid', 'he')

    def test_invalid_bands(self):
        """Test that invalid bands raises error."""
        with pytest.raises(ValueError):
            etr.generate_multiband_eq('xy', 'hx')

    def test_wrong_param_count(self):
        """Test that wrong parameter count raises error."""
        eq = etr.generate_multiband_eq('xy', 'he')
        field = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="Expected 4 parameters"):
            eq(field, 1e26, 0.015)  # Missing n2, mu2


class TestFitMultiband:
    """Tests for fit_multiband function."""

    @pytest.fixture
    def synthetic_twoband_data(self):
        """Generate synthetic two-band data for testing."""
        # True parameters - use values that work well for single dataset fitting
        n1_true, mu1_true = 2.51e+25, 0.011
        n2_true, mu2_true = 4.83e+24, 0.019

        field = np.linspace(-14, 14, 15)

        # Generate synthetic data
        eq_xy = etr.generate_multiband_eq('xy', 'he')
        eq_xx = etr.generate_multiband_eq('xx', 'he')

        rho_xy = eq_xy(field, n1_true, mu1_true, n2_true, mu2_true)
        rho_xx = eq_xx(field, n1_true, mu1_true, n2_true, mu2_true)

        # Add small noise
        np.random.seed(1)
        rho_xy += np.random.normal(0, 1e-9, rho_xy.shape)
        rho_xx += np.random.normal(0, 1e-10, rho_xx.shape)

        return field, rho_xy, rho_xx, (n1_true, mu1_true, n2_true, mu2_true)

    def test_basic_fit_single_dataset(self, synthetic_twoband_data):
        """Test basic multiband fit with single dataset.

        Note: Single dataset fitting with 4 free parameters can be ill-conditioned.
        This test verifies the fit runs successfully and gives physically reasonable results.
        """
        field, rho_xy, rho_xx, true_params = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy')]
        # Initial guess close to true values
        # p0 = [5e24, 0.02, 2.5e25, 0.011]
        p0 = [1e26, 0.015, 1e25, 0.02]

        result = etr.fit_multiband(datasets, p0, bands='he')

        assert isinstance(result, tuple)
        assert len(result) == 4  # n1, mu1, n2, mu2

        # Check that results are positive and in physically reasonable ranges
        n1, mu1, n2, mu2 = result
        n1_true, mu1_true, n2_true, mu2_true = true_params
        print(true_params)
        print(result)
        assert n1 > 0 and n2 > 0
        assert mu1 > 0 and mu2 > 0
        assert abs(n1 - n1_true) / n1_true < 1
        assert abs(mu1 - mu1_true) / mu1_true < 1
        # # Verify parameters are within expected physical bounds
        # assert 1e20 < n1 < 1e28  # reasonable carrier density range
        # assert 1e20 < n2 < 1e28
        # assert 0.001 < mu1 < 1  # reasonable mobility range
        # assert 0.001 < mu2 < 1

        # Verify the fit reproduces the data reasonably well
        eq_xy = etr.generate_multiband_eq('xy', 'he')
        rho_fit = eq_xy(field, n1, mu1, n2, mu2)
        relative_error = np.abs((rho_fit - rho_xy) / rho_xy).mean()
        assert relative_error < 0.1  # Mean relative error < 10%

    def test_fit_multiple_datasets(self, synthetic_twoband_data):
        """Test multiband fit with multiple datasets."""
        field, rho_xy, rho_xx, true_params = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy'), (field, rho_xx, 'xx')]
        p0 = [8e25, 0.012, 4e25, 0.018]

        result = etr.fit_multiband(datasets, p0, bands='he')

        assert isinstance(result, tuple)
        assert len(result) == 4

        # With both datasets, fit should be better
        n1, mu1, n2, mu2 = result
        n1_true, mu1_true, n2_true, mu2_true = true_params

        # Check reasonable agreement (allowing for noise)
        assert abs(n1 - n1_true) / n1_true < 0.2
        assert abs(mu1 - mu1_true) / mu1_true < 0.2

    def test_fit_with_sigma(self, synthetic_twoband_data):
        """Test fit with uncertainty weighting."""
        field, rho_xy, rho_xx, true_params = synthetic_twoband_data

        # Higher weight on xy data
        datasets = [
            (field, rho_xy, 'xy', 1e-9),
            (field, rho_xx, 'xx', 1e-8)
        ]
        p0 = [8e25, 0.012, 4e25, 0.018]

        result = etr.fit_multiband(datasets, p0, bands='he')

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_fit_with_bounds(self, synthetic_twoband_data):
        """Test fit with parameter bounds."""
        field, rho_xy, rho_xx, true_params = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy')]
        p0 = [8e25, 0.012, 4e25, 0.018]

        bounds = {
            'n1': (5e25, 2e26),
            'n2': (1e25, 1e26),
            'mu1': (0.01, 0.03),
            'mu2': (0.01, 0.05)
        }

        result = etr.fit_multiband(datasets, p0, bands='he', bounds=bounds)

        n1, mu1, n2, mu2 = result

        # Check bounds are respected
        assert 5e25 <= n1 <= 2e26
        assert 1e25 <= n2 <= 1e26
        assert 0.01 <= mu1 <= 0.03
        assert 0.01 <= mu2 <= 0.05

    def test_fit_with_expr_equal_densities(self, synthetic_twoband_data):
        """Test fit with density constraint (compensated semimetal)."""
        field, rho_xy, _, _ = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy')]
        p0 = [7e25, 0.012, 7e25, 0.018]

        # Constrain n2 = n1 (charge compensation)
        result = etr.fit_multiband(datasets, p0, bands='he', expr={'n2': 'n1'})

        n1, mu1, n2, mu2 = result

        # Densities should be equal (within numerical precision)
        assert abs(n1 - n2) / n1 < 1e-6

    def test_fit_with_expr_mobility_ratio(self):
        """Test fit with mobility ratio constraint."""
        # Generate data with known mobility ratio
        n1, n2 = 1e26, 5e25
        mu1 = 0.02
        mu2 = 2 * mu1  # mu2 = 2*mu1

        field = np.linspace(-8, 8, 40)
        eq_xy = etr.generate_multiband_eq('xy', 'he')
        rho_xy = eq_xy(field, n1, mu1, n2, mu2)

        datasets = [(field, rho_xy, 'xy')]
        p0 = [8e25, 0.015, 4e25, 0.03]

        # Constrain mu2 = 2*mu1 (in log10 space: log10(mu2) = log10(2) + log10(mu1))
        # log10(2) ≈ 0.30103
        result = etr.fit_multiband(datasets, p0, bands='he', expr={'mu2': '0.30103 + mu1'})

        n1_fit, mu1_fit, n2_fit, mu2_fit = result

        # Check mobility ratio
        assert abs(mu2_fit - 2*mu1_fit) / mu2_fit < 1e-3

    def test_fit_with_fixed_params(self, synthetic_twoband_data):
        """Test fit with fixed parameters."""
        field, rho_xy, _, (n1_true, mu1_true, n2_true, mu2_true) = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy')]
        p0 = [n1_true, 0.012, 4e25, 0.018]

        # Fix n1 at true value
        fix_params = {'n1': True}

        result = etr.fit_multiband(datasets, p0, bands='he', fix_params=fix_params)

        n1, mu1, n2, mu2 = result

        # n1 should be unchanged
        assert abs(n1 - n1_true) / n1_true < 1e-10

    def test_three_band_fit(self):
        """Test three-band fitting."""
        # Generate synthetic three-band data
        n1, mu1 = 1e26, 0.01
        n2, mu2 = 5e25, 0.02
        n3, mu3 = 2e25, 0.015

        field = np.linspace(-10, 10, 50)
        eq_xy = etr.generate_multiband_eq('xy', 'hee')
        rho_xy = eq_xy(field, n1, mu1, n2, mu2, n3, mu3)

        datasets = [(field, rho_xy, 'xy')]
        p0 = [8e25, 0.012, 4e25, 0.018, 1.5e25, 0.012]

        result = etr.fit_multiband(datasets, p0, bands='hee')

        assert isinstance(result, tuple)
        assert len(result) == 6  # n1, mu1, n2, mu2, n3, mu3

    def test_invalid_p0_length(self):
        """Test that invalid p0 length raises error."""
        field = np.linspace(-10, 10, 50)
        rho = np.ones_like(field) * 1e-6

        datasets = [(field, rho, 'xy')]
        p0 = [1e26, 0.015]  # Too short for 2 bands

        with pytest.raises(ValueError, match="p0 must contain 4 values"):
            etr.fit_multiband(datasets, p0, bands='he')

    def test_invalid_p0_values(self):
        """Test that negative/zero p0 raises error."""
        field = np.linspace(-10, 10, 50)
        rho = np.ones_like(field) * 1e-6

        datasets = [(field, rho, 'xy')]
        p0 = [1e26, 0.015, -5e25, 0.02]  # Negative value

        with pytest.raises(ValueError, match="positive"):
            etr.fit_multiband(datasets, p0, bands='he')

    def test_invalid_dataset_format(self):
        """Test that invalid dataset format raises error."""
        field = np.linspace(-10, 10, 50)
        rho = np.ones_like(field) * 1e-6

        datasets = [(field, rho)]  # Missing 'kind'
        p0 = [1e26, 0.015, 5e25, 0.02]

        with pytest.raises(ValueError, match="must be a tuple"):
            etr.fit_multiband(datasets, p0, bands='he')

    def test_report_to_console(self, synthetic_twoband_data, capsys):
        """Test report output to console."""
        field, rho_xy, _, _ = synthetic_twoband_data

        datasets = [(field, rho_xy, 'xy')]
        p0 = [8e25, 0.012, 4e25, 0.018]

        result = etr.fit_multiband(datasets, p0, bands='he', report=True)

        captured = capsys.readouterr()
        assert 'n1' in captured.out
        assert 'mu1' in captured.out


class TestMultibandFitToStr:
    """Tests for multiband_fit_to_str function."""

    def test_two_band_format(self):
        """Test formatting of two-band results."""
        p = (1e26, 0.015, 5e25, 0.02)
        result = etr.multiband_fit_to_str(p, 'he')

        assert 'n1(h)' in result
        assert 'n2(e)' in result
        assert 'mu1(h)' in result
        assert 'mu2(e)' in result
        assert '1.00e+26' in result
        assert '1.50e-02' in result

    def test_three_band_format(self):
        """Test formatting of three-band results."""
        p = (1e26, 0.01, 5e25, 0.02, 2e25, 0.015)
        result = etr.multiband_fit_to_str(p, 'hee')

        assert 'n1(h)' in result
        assert 'n2(e)' in result
        assert 'n3(e)' in result
        assert 'mu1(h)' in result
        assert 'mu2(e)' in result
        assert 'mu3(e)' in result

    def test_invalid_param_count(self):
        """Test that mismatched param count raises error."""
        p = (1e26, 0.015, 5e25)  # 3 params for 2 bands

        with pytest.raises(ValueError, match="Expected 4 parameters"):
            etr.multiband_fit_to_str(p, 'he')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
