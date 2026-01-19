"""Tests for MeasurementFile class."""

import pytest
from pathlib import Path
from decimal import Decimal
from medapy.collection.mfile import MeasurementFile, ContactPair
from medapy.collection.parameter import ParameterDefinition, DefinitionsLoader


class TestMeasurementFileCreation:
    """Test MeasurementFile creation and initialization."""

    @pytest.fixture
    def temp_definitions(self):
        """Create basic parameter definitions for testing."""
        return [
            ParameterDefinition(
                name_id="temperature",
                long_names=["temperature"],
                short_names=["T"],
                units=["K"]
            ),
            ParameterDefinition(
                name_id="magnetic_field",
                long_names=["field"],
                short_names=["B"],
                units=["T"]
            ),
        ]

    def test_create_from_path_with_definitions(self, temp_definitions):
        """Test creating MeasurementFile with parameter definitions."""
        mfile = MeasurementFile(
            path="test_file.csv",
            parameters=temp_definitions
        )
        assert mfile.path == Path("test_file.csv")
        assert mfile.name == "test_file.csv"
        assert "temperature" in mfile.param_definitions
        assert "magnetic_field" in mfile.param_definitions

    def test_create_with_custom_separator(self, temp_definitions):
        """Test creating MeasurementFile with custom separator."""
        mfile = MeasurementFile(
            path="test-file.csv",
            parameters=temp_definitions,
            separator="-"
        )
        assert mfile.separator == "-"


class TestMeasurementFilenameParsing:
    """Test parsing filenames to extract parameters and contacts."""

    @pytest.fixture
    def definitions_loader(self):
        """Use default parameter definitions."""
        return DefinitionsLoader()

    def test_parse_simple_filename(self, definitions_loader):
        """Test parsing filename with simple parameters."""
        mfile = MeasurementFile(
            path="sample_T=4.2K_B=0T.csv",
            parameters=definitions_loader.get_all()
        )
        assert "temperature" in mfile.parameters
        assert mfile.parameters["temperature"].state.value == Decimal('4.2')
        assert mfile.parameters["temperature"].state.unit == 'K'

        assert "magnetic_field" in mfile.parameters
        assert mfile.parameters["magnetic_field"].state.value == Decimal('0')

    def test_parse_contacts_from_filename(self, definitions_loader):
        """Test extracting contact pairs from filename."""
        mfile = MeasurementFile(
            path="sample_I1-5(10mA)_V20-21_T=4.2K.csv",
            parameters=definitions_loader.get_all()
        )
        assert len(mfile.contact_pairs) == 2
        assert mfile.contact_pairs[0] == ContactPair(1, 5, 'I', Decimal('0.01'))
        assert mfile.contact_pairs[1] == ContactPair(20, 21, 'V', None)

    def test_parse_sweep_parameter(self, definitions_loader):
        """Test parsing sweep parameters."""
        mfile = MeasurementFile(
            path="sample_sweepField_T=4.2K.csv",
            parameters=definitions_loader.get_all()
        )
        assert "magnetic_field" in mfile.parameters
        assert mfile.parameters["magnetic_field"].state.is_swept is True

    def test_parse_range_parameter(self, definitions_loader):
        """Test parsing range parameters."""
        mfile = MeasurementFile(
            path="sample_B-14to14T_T=4.2K.csv",
            parameters=definitions_loader.get_all()
        )
        assert "magnetic_field" in mfile.parameters
        field = mfile.parameters["magnetic_field"]
        assert field.state.is_swept is True
        assert field.state.min_val == Decimal('-14')
        assert field.state.max_val == Decimal('14')


class TestMeasurementFileCheckContacts:
    """Test check_contacts method with different matching modes."""

    @pytest.fixture
    def mfile_with_contacts(self, sample_filenames):
        """Create MeasurementFile with contacts."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path=sample_filenames[0],  # "sample_I1-5(10mA)_V20-21_sweepField_T=4.2K_Rxx.csv"
            parameters=loader.get_all()
        )

    def test_check_single_contact_match(self, mfile_with_contacts):
        """Test checking for single contact that matches."""
        # File has I1-5 and V20-21, check for single contact in first pair
        # This won't match because I1-5 is a pair, not a single contact
        assert mfile_with_contacts.check_contacts(1) is False

    def test_check_contact_pair_match(self, mfile_with_contacts):
        """Test checking for contact pair."""
        # Check for the exact pair
        assert mfile_with_contacts.check_contacts((1, 5)) is True
        assert mfile_with_contacts.check_contacts((20, 21)) is True

    def test_check_contact_pair_no_match(self, mfile_with_contacts):
        """Test checking for contact pair that doesn't exist."""
        assert mfile_with_contacts.check_contacts((2, 3)) is False

    def test_check_contact_string(self, mfile_with_contacts):
        """Test checking contact using string."""
        assert mfile_with_contacts.check_contacts("I1-5(10mA)") is True
        assert mfile_with_contacts.check_contacts("V20-21") is True

    def test_check_multiple_contacts(self, mfile_with_contacts):
        """Test checking multiple contacts (all must match)."""
        assert mfile_with_contacts.check_contacts([(1, 5), (20, 21)]) is True
        assert mfile_with_contacts.check_contacts([(1, 5), (2, 3)]) is False


class TestMeasurementFileCheckParameter:
    """Test check_parameter method with fixed values and ranges."""

    @pytest.fixture
    def mfile_fixed(self):
        """MeasurementFile with fixed parameters."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_T=4.2K_B=0T.csv",
            parameters=loader.get_all()
        )

    @pytest.fixture
    def mfile_swept(self):
        """MeasurementFile with swept parameter."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_B-14to14T_T=4.2K.csv",
            parameters=loader.get_all()
        )

    def test_check_fixed_parameter_exact_value(self, mfile_fixed):
        """Test checking fixed parameter with exact value."""
        assert mfile_fixed.check_parameter("temperature", 4.2) is True
        assert mfile_fixed.check_parameter("temperature", 4.0) is False

    def test_check_fixed_parameter_with_range(self, mfile_fixed):
        """Test checking fixed parameter within range."""
        # Temperature is 4.2K
        assert mfile_fixed.check_parameter("temperature", (4.0, 5.0)) is True
        assert mfile_fixed.check_parameter("temperature", (5.0, 10.0)) is False

    def test_check_fixed_parameter_open_boundaries(self, mfile_fixed):
        """Test checking with None for open boundaries."""
        # Temperature is 4.2K
        assert mfile_fixed.check_parameter("temperature", (None, 5.0)) is True
        assert mfile_fixed.check_parameter("temperature", (5.0, None)) is False

    def test_check_swept_parameter_exact_range(self, mfile_swept):
        """Test checking swept parameter with exact range match."""
        # Field is swept from -14 to 14 T
        assert mfile_swept.check_parameter(
            "magnetic_field", (-14, 14), swept=True, exact_sweep=True
        ) is True
        assert mfile_swept.check_parameter(
            "magnetic_field", (-10, 10), swept=True, exact_sweep=True
        ) is False

    def test_check_swept_parameter_contains_range(self, mfile_swept):
        """Test checking swept parameter with non-exact range."""
        # Field is swept from -14 to 14 T
        # Check if file sweep is contained within specified range
        assert mfile_swept.check_parameter(
            "magnetic_field", (-15, 15), swept=True, exact_sweep=False
        ) is True
        # File sweep (-14, 14) is NOT contained in (-10, 10)
        assert mfile_swept.check_parameter(
            "magnetic_field", (-10, 10), swept=True, exact_sweep=False
        ) is False


class TestMeasurementFileCheck:
    """Test the main check() method with various filters."""

    @pytest.fixture
    def mfile(self):
        """MeasurementFile for general testing."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_I1-5(10mA)_V20-21_B-14to14T_T=4.2K_Rxx.csv",
            parameters=loader.get_all()
        )

    def test_check_with_contacts(self, mfile):
        """Test check() with contacts filter."""
        assert mfile.check(contacts=(1, 5)) is True
        assert mfile.check(contacts=(2, 3)) is False

    def test_check_with_polarization(self, mfile):
        """Test check() with polarization filter."""
        assert mfile.check(polarization='I') is True
        assert mfile.check(polarization='V') is True

    def test_check_with_sweeps(self, mfile):
        """Test check() with sweeps filter."""
        assert mfile.check(sweeps="magnetic_field") is True
        assert mfile.check(sweeps="temperature") is False

    def test_check_with_parameter_exact(self, mfile):
        """Test check() with parameter exact value."""
        assert mfile.check(temperature=4.2) is True
        assert mfile.check(temperature=300) is False

    def test_check_with_parameter_range(self, mfile):
        """Test check() with parameter range."""
        assert mfile.check(temperature=(4.0, 5.0)) is True
        assert mfile.check(temperature=(5.0, 10.0)) is False

    def test_check_with_sweep_parameter(self, mfile):
        """Test check() with swept parameter."""
        assert mfile.check(magnetic_field_sweep=(-14, 14)) is True
        assert mfile.check(magnetic_field_sweep=(-10, 10)) is False

    def test_check_multiple_conditions_all_mode(self, mfile):
        """Test check() with multiple conditions using 'all' mode."""
        assert mfile.check(
            mode='all',
            contacts=(1, 5),
            temperature=4.2,
            sweeps="magnetic_field"
        ) is True

        # One condition fails
        assert mfile.check(
            mode='all',
            contacts=(1, 5),
            temperature=300,
            sweeps="magnetic_field"
        ) is False

    def test_check_multiple_conditions_any_mode(self, mfile):
        """Test check() with multiple conditions using 'any' mode."""
        # At least one condition matches
        assert mfile.check(
            mode='any',
            contacts=(1, 5),
            temperature=300
        ) is True

        # No conditions match
        assert mfile.check(
            mode='any',
            contacts=(2, 3),
            temperature=300
        ) is False

    def test_check_invalid_mode(self, mfile):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'all' or 'any'"):
            mfile.check(mode='invalid')


class TestMeasurementFileCheckName:
    """Test check_name method for filename pattern matching."""

    @pytest.fixture
    def mfile(self):
        """MeasurementFile for name checking."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_device_A_T=4.2K_Rxx.csv",
            parameters=loader.get_all()
        )

    def test_check_name_substring(self, mfile):
        """Test checking for substring in filename."""
        assert mfile.check_name("device_A") is True
        assert mfile.check_name("Rxx") is True
        assert mfile.check_name("device_B") is False

    def test_check_name_regex(self, mfile):
        """Test checking with regex pattern."""
        assert mfile.check_name(r"device_[A-Z]") is True
        assert mfile.check_name(r"T=\d+\.\d+K") is True

    def test_check_name_multiple_patterns(self, mfile):
        """Test checking multiple patterns (all must match)."""
        assert mfile.check_name(["device_A", "Rxx"]) is True
        assert mfile.check_name(["device_A", "Rxy"]) is False


class TestMeasurementFileCheckSweeps:
    """Test sweep checking functionality."""

    @pytest.fixture
    def mfile_one_sweep(self):
        """MeasurementFile with one swept parameter."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_B-14to14T_T=4.2K.csv",
            parameters=loader.get_all()
        )

    def test_check_sweep_exists(self, mfile_one_sweep):
        """Test checking if parameter is swept."""
        assert mfile_one_sweep.check_sweep("magnetic_field") is True
        assert mfile_one_sweep.check_sweep("temperature") is False

    def test_check_sweep_with_direction(self, mfile_one_sweep):
        """Test checking sweep with direction."""
        # Field swept -14 to 14 (increasing)
        assert mfile_one_sweep.check_sweep("magnetic_field", "inc") is True
        assert mfile_one_sweep.check_sweep("magnetic_field", "dec") is False

    def test_check_sweep_nonexistent_parameter(self, mfile_one_sweep):
        """Test checking sweep on non-existent parameter."""
        assert mfile_one_sweep.check_sweep("nonexistent") is False


class TestMeasurementFileCheckPolarization:
    """Test polarization checking."""

    @pytest.fixture
    def mfile(self):
        """MeasurementFile with polarized contacts."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_I1-5(10mA)_V20-21.csv",
            parameters=loader.get_all()
        )

    def test_check_polarization_current(self, mfile):
        """Test checking for current polarization."""
        assert mfile.check_polarization('I') is True

    def test_check_polarization_voltage(self, mfile):
        """Test checking for voltage polarization."""
        assert mfile.check_polarization('V') is True


class TestMeasurementFileParameterMethods:
    """Test parameter getter and setter methods."""

    @pytest.fixture
    def mfile(self):
        """MeasurementFile for parameter manipulation."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_T=4.2K.csv",
            parameters=loader.get_all()
        )

    def test_get_parameter(self, mfile):
        """Test getting a parameter."""
        param = mfile.get_parameter("temperature")
        assert param.state.value == Decimal('4.2')

    def test_get_nonexistent_parameter(self, mfile):
        """Test that getting non-existent parameter raises ValueError."""
        with pytest.raises(ValueError, match="not defined for file"):
            mfile.get_parameter("nonexistent")

    def test_set_parameter_fixed(self, mfile):
        """Test setting a parameter to fixed value."""
        mfile.set_parameter_fixed("temperature", 300)
        assert mfile.parameters["temperature"].state.value == Decimal('300')
        assert mfile.parameters["temperature"].state.is_swept is False

    def test_set_parameter_swept(self, mfile):
        """Test setting a parameter as swept."""
        # Use temperature since it exists in the file
        mfile.set_parameter_swept("temperature", 1.8, 300)
        temp = mfile.parameters["temperature"]
        assert temp.state.is_swept is True
        assert temp.state.min_val == Decimal('1.8')
        assert temp.state.max_val == Decimal('300')


class TestMeasurementFileIntegration:
    """Integration tests with realistic filenames."""

    @pytest.fixture
    def loader(self):
        """Parameter definitions loader."""
        return DefinitionsLoader()

    def test_realistic_filename_1(self, loader, sample_filenames):
        """Test parsing realistic filename 1."""
        # "sample_I1-5(10mA)_V20-21_sweepField_T=4.2K_Rxx.csv"
        mfile = MeasurementFile(
            path=sample_filenames[0],
            parameters=loader.get_all()
        )

        # Check contacts
        assert len(mfile.contact_pairs) == 2
        assert mfile.check_contacts((1, 5)) is True
        assert mfile.check_contacts((20, 21)) is True

        # Check parameters
        assert mfile.check_parameter("temperature", 4.2) is True
        assert mfile.check_sweep("magnetic_field") is True

    def test_realistic_filename_2(self, loader, sample_filenames):
        """Test parsing realistic filename 2."""
        # "sample_V2-3_B-14to14T_T=1.8K_Rxy.csv"
        mfile = MeasurementFile(
            path=sample_filenames[1],
            parameters=loader.get_all()
        )

        # Check contacts
        assert mfile.check_contacts((2, 3)) is True

        # Check parameters
        assert mfile.check_parameter("temperature", 1.8) is True
        assert mfile.check_parameter("magnetic_field", (-14, 14), swept=True) is True

    def test_complex_filtering(self, loader, sample_filenames):
        """Test complex filtering with multiple conditions."""
        mfile = MeasurementFile(
            path=sample_filenames[0],
            parameters=loader.get_all()
        )

        # All conditions match
        assert mfile.check(
            mode='all',
            contacts=[(1, 5), (20, 21)],
            polarization='I',
            sweeps="magnetic_field",
            temperature=(4.0, 5.0)
        ) is True

        # Some conditions don't match
        assert mfile.check(
            mode='all',
            contacts=(1, 5),
            temperature=300
        ) is False

        # At least one matches with 'any' mode
        assert mfile.check(
            mode='any',
            contacts=(1, 5),
            temperature=300
        ) is True


class TestMeasurementFileContactManipulation:
    """Test add_contacts, update_contacts, and drop_contacts methods."""

    @pytest.fixture
    def mfile_no_contacts(self):
        """MeasurementFile without contacts."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_T=4.2K.csv",
            parameters=loader.get_all()
        )

    @pytest.fixture
    def mfile_with_contacts(self):
        """MeasurementFile with existing contacts."""
        loader = DefinitionsLoader()
        return MeasurementFile(
            path="sample_I1-2(2uA)_V3-4_T=4.2K.csv",
            parameters=loader.get_all()
        )

    # Tests for add_contacts
    def test_add_single_contact(self, mfile_no_contacts):
        """Test adding a single contact."""
        mfile_no_contacts.add_contacts(5)
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(5)

    def test_add_contact_pair(self, mfile_no_contacts):
        """Test adding a contact pair."""
        mfile_no_contacts.add_contacts((1, 2))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2)

    def test_add_contact_with_polarization(self, mfile_no_contacts):
        """Test adding contact with polarization."""
        mfile_no_contacts.add_contacts((1, 2, 'I'))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2, 'I', None)

    def test_add_contact_with_polarization_and_magnitude(self, mfile_no_contacts):
        """Test adding contact with polarization and magnitude."""
        mfile_no_contacts.add_contacts((1, 2, 'I', Decimal('0.000002')))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2, 'I', Decimal('0.000002'))

    def test_add_contact_pair_object(self, mfile_no_contacts):
        """Test adding a ContactPair object directly."""
        pair = ContactPair(3, 4, 'V', Decimal('0.001'))
        mfile_no_contacts.add_contacts(pair)
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == pair

    def test_add_multiple_contacts(self, mfile_no_contacts):
        """Test adding multiple contacts at once."""
        mfile_no_contacts.add_contacts([(1, 2), (3, 4, 'V'), 5])
        assert len(mfile_no_contacts.contact_pairs) == 3
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2)
        assert mfile_no_contacts.contact_pairs[1] == ContactPair(3, 4, 'V')
        assert mfile_no_contacts.contact_pairs[2] == ContactPair(5)

    def test_add_duplicate_contact_raises_error(self, mfile_with_contacts):
        """Test that adding exact duplicate contact raises ValueError."""
        # File already has I1-2(2uA), add exact duplicate
        with pytest.raises(ValueError, match="Contact pair .* already exists"):
            mfile_with_contacts.add_contacts((1, 2, 'I', Decimal('0.000002')))

    def test_add_same_contacts_different_polarization_allowed(self, mfile_with_contacts):
        """Test that same contact numbers with different polarization is allowed."""
        # File has I1-2(2uA), can add V1-2 (different polarization)
        initial_count = len(mfile_with_contacts.contact_pairs)
        mfile_with_contacts.add_contacts((1, 2, 'V', Decimal('0.001')))
        assert len(mfile_with_contacts.contact_pairs) == initial_count + 1

        # Can also add unpolarized (1, 2)
        mfile_with_contacts.add_contacts((1, 2))
        assert len(mfile_with_contacts.contact_pairs) == initial_count + 2

    # Tests for update_contacts
    def test_update_contact_polarization(self, mfile_with_contacts):
        """Test updating contact polarization only."""
        # File has I1-2(2uA), update polarization to V
        # Note: Not specifying magnitude sets it to None
        mfile_with_contacts.update_contacts((1, 2, 'V'))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.first_contact == 1
        assert pair.second_contact == 2
        assert pair.polarization.value == 'V'
        # Magnitude becomes None when not specified in update
        assert pair.magnitude is None

    def test_update_polarization_preserving_magnitude(self, mfile_with_contacts):
        """Test updating polarization while preserving magnitude."""
        # File has I1-2(2uA), update to V1-2(2uA)
        original_magnitude = mfile_with_contacts.contact_pairs[0].magnitude
        mfile_with_contacts.update_contacts((1, 2, 'V', original_magnitude))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.polarization.value == 'V'
        assert pair.magnitude == Decimal('0.000002')

    def test_update_contact_magnitude(self, mfile_with_contacts):
        """Test updating contact magnitude."""
        # File has I1-2(2uA), update magnitude
        mfile_with_contacts.update_contacts((1, 2, 'I', Decimal('0.000005')))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.polarization.value == 'I'
        assert pair.magnitude == Decimal('0.000005')

    def test_update_removes_polarization(self, mfile_with_contacts):
        """Test updating to remove polarization."""
        # File has I1-2(2uA), update to just (1, 2)
        mfile_with_contacts.update_contacts((1, 2))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.first_contact == 1
        assert pair.second_contact == 2
        assert pair.polarization is None
        assert pair.magnitude is None

    def test_update_contact_with_none_magnitude(self, mfile_with_contacts):
        """Test updating to remove magnitude but keep polarization."""
        # File has I1-2(2uA), update to I1-2 (no magnitude)
        mfile_with_contacts.update_contacts((1, 2, 'I', None))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.polarization.value == 'I'
        assert pair.magnitude is None

    def test_update_multiple_contacts(self, mfile_with_contacts):
        """Test updating multiple contacts at once."""
        # File has I1-2(2uA) and V3-4
        mfile_with_contacts.update_contacts([
            (1, 2, 'V', Decimal('0.001')),
            (3, 4, 'I', Decimal('0.00001'))
        ])
        assert mfile_with_contacts.contact_pairs[0].polarization.value == 'V'
        assert mfile_with_contacts.contact_pairs[0].magnitude == Decimal('0.001')
        assert mfile_with_contacts.contact_pairs[1].polarization.value == 'I'
        assert mfile_with_contacts.contact_pairs[1].magnitude == Decimal('0.00001')

    def test_update_nonexistent_contact_raises_error(self, mfile_with_contacts):
        """Test that updating non-existent contact raises ValueError."""
        with pytest.raises(ValueError, match="Contact pair .* not found"):
            mfile_with_contacts.update_contacts((5, 6))

    def test_update_matches_by_contacts_only(self, mfile_with_contacts):
        """Test that update matches by contact numbers, not polarization."""
        # File has I1-2(2uA), should match even if we specify different polarization
        mfile_with_contacts.update_contacts((1, 2, 'V'))
        pair = mfile_with_contacts.contact_pairs[0]
        assert pair.polarization.value == 'V'

    # Tests for drop_contacts
    def test_drop_single_contact(self, mfile_no_contacts):
        """Test dropping a single contact."""
        mfile_no_contacts.add_contacts([5, (1, 2)])
        mfile_no_contacts.drop_contacts(5)
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2)

    def test_drop_contact_pair(self, mfile_with_contacts):
        """Test dropping a contact pair."""
        # File has I1-2(2uA) and V3-4
        initial_count = len(mfile_with_contacts.contact_pairs)
        mfile_with_contacts.drop_contacts((1, 2))
        assert len(mfile_with_contacts.contact_pairs) == initial_count - 1
        assert not mfile_with_contacts.check_contacts((1, 2))

    def test_drop_by_contacts_and_polarization(self, mfile_no_contacts):
        """Test dropping by contacts and polarization."""
        # Add two pairs with same contacts but different polarization
        mfile_no_contacts.add_contacts([(1, 2, 'I'), (1, 2, 'V')])
        assert len(mfile_no_contacts.contact_pairs) == 2

        # Drop only the voltage pair
        mfile_no_contacts.drop_contacts((1, 2, 'V'))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0].polarization.value == 'I'

    def test_drop_exact_match(self, mfile_no_contacts):
        """Test dropping with exact match (4-tuple)."""
        mfile_no_contacts.add_contacts([
            (1, 2, 'I', Decimal('0.000001')),
            (1, 2, 'I', Decimal('0.000002'))
        ])
        assert len(mfile_no_contacts.contact_pairs) == 2

        # Drop exact match
        mfile_no_contacts.drop_contacts((1, 2, 'I', Decimal('0.000001')))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0].magnitude == Decimal('0.000002')

    def test_drop_multiple_contacts(self, mfile_with_contacts):
        """Test dropping multiple contacts at once."""
        # File has I1-2(2uA) and V3-4
        mfile_with_contacts.drop_contacts([(1, 2), (3, 4)])
        assert len(mfile_with_contacts.contact_pairs) == 0

    def test_drop_all_contacts(self, mfile_with_contacts):
        """Test dropping all contacts with 'all' keyword."""
        assert len(mfile_with_contacts.contact_pairs) > 0
        mfile_with_contacts.drop_contacts('all')
        assert len(mfile_with_contacts.contact_pairs) == 0

    def test_drop_nonexistent_contact_raises_error(self, mfile_with_contacts):
        """Test that dropping non-existent contact raises ValueError."""
        with pytest.raises(ValueError, match="Contact .* not found"):
            mfile_with_contacts.drop_contacts((5, 6))

    def test_drop_first_match_if_ambiguous(self, mfile_no_contacts):
        """Test that drop removes first match when multiple exist."""
        # Add two I1-2 pairs with different magnitudes
        mfile_no_contacts.add_contacts([
            (1, 2, 'I', Decimal('0.000001')),
            (1, 2, 'I', Decimal('0.000002'))
        ])

        # Drop by (1, 2) should remove first match
        mfile_no_contacts.drop_contacts((1, 2))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0].magnitude == Decimal('0.000002')

    # Integration tests
    def test_add_update_drop_workflow(self, mfile_no_contacts):
        """Test complete workflow of adding, updating, and dropping contacts."""
        # Add contacts
        mfile_no_contacts.add_contacts([(1, 2, 'I', Decimal('0.000001')), (3, 4)])
        assert len(mfile_no_contacts.contact_pairs) == 2

        # Update first contact
        mfile_no_contacts.update_contacts((1, 2, 'V', Decimal('0.002')))
        assert mfile_no_contacts.contact_pairs[0].polarization.value == 'V'
        assert mfile_no_contacts.contact_pairs[0].magnitude == Decimal('0.002')

        # Update second contact
        mfile_no_contacts.update_contacts((3, 4, 'I'))
        assert mfile_no_contacts.contact_pairs[1].polarization.value == 'I'

        # Drop one contact
        mfile_no_contacts.drop_contacts((1, 2))
        assert len(mfile_no_contacts.contact_pairs) == 1
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(3, 4, 'I', None)

        # Drop remaining
        mfile_no_contacts.drop_contacts('all')
        assert len(mfile_no_contacts.contact_pairs) == 0

    def test_contact_manipulation_preserves_order(self, mfile_no_contacts):
        """Test that contact manipulation preserves order."""
        mfile_no_contacts.add_contacts([(1, 2), (3, 4), (5, 6)])

        # Update middle contact
        mfile_no_contacts.update_contacts((3, 4, 'I'))

        # Check order is preserved
        assert mfile_no_contacts.contact_pairs[0] == ContactPair(1, 2)
        assert mfile_no_contacts.contact_pairs[1] == ContactPair(3, 4, 'I')
        assert mfile_no_contacts.contact_pairs[2] == ContactPair(5, 6)
