"""Tests for Parameter and ParameterDefinition classes."""

import pytest
from decimal import Decimal
from types import MappingProxyType
from medapy.collection.parameter import (
    Parameter,
    ParameterDefinition,
    SweepDirection,
    DefinitionsLoader,
    ParameterState,
)


class TestParameterDefinition:
    """Test ParameterDefinition class."""

    def test_create_simple_definition(self):
        """Test creating a basic parameter definition."""
        pdef = ParameterDefinition(
            name_id="temperature",
            long_names=["temperature", "temp"],
            short_names=["T"],
            units=["K", "C"]
        )
        assert pdef.name_id == "temperature"
        assert "temperature" in pdef.long_names
        assert "T" in pdef.short_names
        assert "K" in pdef.units

    def test_long_and_short_names_must_exist(self):
        """Test that at least one of long/short names must exist."""
        with pytest.raises(ValueError, match="Long and short names cannot be empty"):
            ParameterDefinition(
                name_id="test",
                long_names=[],
                short_names=[]
            )

    def test_long_and_short_names_cannot_overlap(self):
        """Test that long and short names cannot overlap."""
        with pytest.raises(ValueError, match="Long and short names overlap"):
            ParameterDefinition(
                name_id="test",
                long_names=["field", "B"],
                short_names=["B"]
            )

    def test_special_values_converted_to_decimal(self):
        """Test that special values are converted to Decimal."""
        pdef = ParameterDefinition(
            name_id="position",
            long_names=["position"],
            special_values={"OOP": 0, "IP": 90}
        )
        assert pdef.special_values["OOP"] == Decimal('0')
        assert pdef.special_values["IP"] == Decimal('90')
        assert isinstance(pdef.special_values, MappingProxyType)

    def test_patterns_compiled(self):
        """Test that patterns are compiled to regex."""
        pdef = ParameterDefinition(
            name_id="temp",
            long_names=["temperature"],
            short_names=["T"],
            units=["K"]
        )
        # Check default patterns exist
        assert 'fixed' in pdef.patterns
        assert 'sweep' in pdef.patterns
        assert 'range' in pdef.patterns

    def test_match_method(self):
        """Test pattern matching."""
        pdef = ParameterDefinition(
            name_id="temp",
            long_names=["temperature"],
            short_names=["T"],
            units=["K"]
        )
        # Match fixed parameter
        m = pdef.match('fixed', 'T=4.2K')
        assert m is not None
        assert m.group(1) == '4.2'  # Value
        assert m.group(2) == 'K'    # Unit

    def test_search_method(self):
        """Test pattern searching."""
        pdef = ParameterDefinition(
            name_id="field",
            long_names=["field"],
            short_names=["B"],
            units=["T"]
        )
        # Search for range parameter (range pattern has $ so must be at end)
        m = pdef.search('range', 'sample_other_B-5to5T')
        assert m is not None


class TestParameter:
    """Test Parameter class."""

    @pytest.fixture
    def temp_definition(self):
        """Temperature parameter definition."""
        return ParameterDefinition(
            name_id="temperature",
            long_names=["temperature", "temp"],
            short_names=["T"],
            units=["K", "mK"]
        )

    @pytest.fixture
    def field_definition(self):
        """Magnetic field parameter definition."""
        return ParameterDefinition(
            name_id="field",
            long_names=["field", "Field"],
            short_names=["B"],
            units=["T", "mT"]
        )

    @pytest.fixture
    def position_definition(self):
        """Position parameter with special values."""
        return ParameterDefinition(
            name_id="position",
            long_names=["position"],
            short_names=["pos"],
            units=["deg"],
            special_values={"OOP": 0, "IP": 90}
        )

    def test_create_parameter(self, temp_definition):
        """Test creating a Parameter instance."""
        param = Parameter(temp_definition)
        assert param.definition == temp_definition
        assert param.state.value is None
        assert param.state.is_swept is False

    def test_parse_fixed_parameter(self, temp_definition):
        """Test parsing fixed parameter value."""
        param = Parameter(temp_definition)
        success = param.parse_fixed("T=4.2K")
        assert success is True
        assert param.state.value == Decimal('4.2')
        assert param.state.unit == 'K'
        assert param.state.is_swept is False

    def test_parse_fixed_parameter_no_unit(self, temp_definition):
        """Test parsing fixed parameter without unit."""
        param = Parameter(temp_definition)
        success = param.parse_fixed("T=300")
        assert success is True
        assert param.state.value == Decimal('300')

    def test_parse_fixed_parameter_no_equals(self, temp_definition):
        """Test parsing fixed parameter without equals sign."""
        param = Parameter(temp_definition)
        success = param.parse_fixed("T4.2K")
        assert success is True
        assert param.state.value == Decimal('4.2')

    def test_parse_range_parameter(self, field_definition):
        """Test parsing range parameter."""
        param = Parameter(field_definition)
        success = param.parse_range("B-14to14T")
        assert success is True
        assert param.state.min_val == Decimal('-14')
        assert param.state.max_val == Decimal('14')
        assert param.state.unit == 'T'
        assert param.state.is_swept is True

    def test_parse_range_handles_order(self, field_definition):
        """Test that range parsing orders values correctly."""
        param = Parameter(field_definition)
        # Provide values in descending order
        success = param.parse_range("B14to-14T")
        assert success is True
        # Should reorder to min, max
        assert param.state.min_val == Decimal('-14')
        assert param.state.max_val == Decimal('14')
        assert param.state.sweep_direction == SweepDirection.DECREASING

    def test_parse_sweep_keyword(self, field_definition):
        """Test parsing sweep keyword without range."""
        param = Parameter(field_definition)
        success = param.parse_sweep("sweepField")
        assert success is True
        assert param.state.is_swept is True
        assert param.state.min_val is None
        assert param.state.max_val is None

    def test_parse_sweep_with_range(self, field_definition):
        """Test that parse_sweep also handles range."""
        param = Parameter(field_definition)
        success = param.parse_sweep("B-5to5T")
        assert success is True
        assert param.state.is_swept is True
        assert param.state.min_val == Decimal('-5')
        assert param.state.max_val == Decimal('5')

    def test_set_fixed_value(self, temp_definition):
        """Test setting fixed value."""
        param = Parameter(temp_definition)
        param.set_fixed(4.2)
        assert param.state.value == Decimal('4.2')
        assert param.state.is_swept is False
        assert param.state.min_val is None
        assert param.state.max_val is None

    def test_set_swept_values(self, field_definition):
        """Test setting swept values."""
        param = Parameter(field_definition)
        param.set_swept(-5, 5)
        assert param.state.is_swept is True
        assert param.state.min_val == Decimal('-5')
        assert param.state.max_val == Decimal('5')
        assert param.state.sweep_direction == SweepDirection.INCREASING
        assert param.state.value is None

    def test_set_swept_determines_direction(self, field_definition):
        """Test that set_swept determines sweep direction."""
        param = Parameter(field_definition)
        # Increasing
        param.set_swept(0, 10)
        assert param.state.sweep_direction == SweepDirection.INCREASING

        # Decreasing (values are swapped)
        param.set_swept(10, 0)
        assert param.state.sweep_direction == SweepDirection.DECREASING
        assert param.state.min_val == Decimal('0')
        assert param.state.max_val == Decimal('10')

    def test_special_values_parsing(self, position_definition):
        """Test parsing special values."""
        param = Parameter(position_definition)
        # Parse special value "OOP"
        value = param.decimal_of("OOP")
        assert value == Decimal('0')

        # Parse special value "IP"
        value = param.decimal_of("IP")
        assert value == Decimal('90')

        # Parse regular numeric value
        value = param.decimal_of("45")
        assert value == Decimal('45')

    def test_parameter_copy(self, temp_definition):
        """Test copying a parameter."""
        param1 = Parameter(temp_definition)
        param1.set_fixed(4.2)
        param1.state.unit = 'K'

        param2 = param1.copy()
        assert param2.state.value == param1.state.value
        assert param2.state.unit == param1.state.unit
        assert param2 is not param1
        assert param2.state is not param1.state

    def test_parameter_update(self, temp_definition):
        """Test updating parameter from another."""
        param1 = Parameter(temp_definition)
        param1.set_fixed(4.2)

        param2 = Parameter(temp_definition)
        param2.set_fixed(300)

        param1.update(param2)
        assert param1.state.value == Decimal('300')

    def test_parameter_str_fixed(self, temp_definition):
        """Test string representation of fixed parameter."""
        param = Parameter(temp_definition)
        param.state.value = Decimal('4.2')
        param.state.unit = 'K'
        result = str(param)
        assert 'T' in result
        assert '4.2' in result
        assert 'K' in result

    def test_parameter_str_swept(self, field_definition):
        """Test string representation of swept parameter."""
        param = Parameter(field_definition)
        param.set_swept(-5, 5)
        param.state.unit = 'T'
        result = str(param)
        assert 'sweep' in result.lower()
        assert '-5' in result
        assert '5' in result
        assert 'T' in result


class TestSweepDirection:
    """Test SweepDirection enum."""

    def test_sweep_direction_values(self):
        """Test SweepDirection enum values."""
        assert SweepDirection.INCREASING.value == 1
        assert SweepDirection.DECREASING.value == -1
        assert SweepDirection.UNDEFINED.value == 0

    @pytest.mark.parametrize("alias,expected", [
        ('inc', SweepDirection.INCREASING),
        ('up', SweepDirection.INCREASING),
        ('increasing', SweepDirection.INCREASING),
        ('dec', SweepDirection.DECREASING),
        ('down', SweepDirection.DECREASING),
        ('decreasing', SweepDirection.DECREASING),
        ('und', SweepDirection.UNDEFINED),
        ('na', SweepDirection.UNDEFINED),
        ('undefined', SweepDirection.UNDEFINED),
    ])
    def test_sweep_direction_aliases(self, alias, expected):
        """Test SweepDirection string aliases."""
        assert SweepDirection(alias) == expected

    def test_sweep_direction_equality_with_string(self):
        """Test that SweepDirection can be compared with strings."""
        assert SweepDirection.INCREASING == 'inc'
        assert SweepDirection.DECREASING == 'down'


class TestDefinitionsLoader:
    """Test DefinitionsLoader class."""

    def test_loader_loads_defaults(self):
        """Test that loader loads default definitions."""
        loader = DefinitionsLoader()
        # Should have loaded from parameter_definitions.json
        assert 'temperature' in loader._definitions
        assert 'magnetic_field' in loader._definitions

    def test_get_definition(self):
        """Test getting a parameter definition."""
        loader = DefinitionsLoader()
        pdef = loader.get_definition('temperature')
        assert isinstance(pdef, ParameterDefinition)
        assert pdef.name_id == 'temperature'
        assert 'T' in pdef.short_names

    def test_get_definition_not_found(self):
        """Test that getting unknown definition raises KeyError."""
        loader = DefinitionsLoader()
        with pytest.raises(KeyError, match="Parameter definition 'unknown' not found"):
            loader.get_definition('unknown')

    def test_get_all_definitions(self):
        """Test getting all definitions."""
        loader = DefinitionsLoader()
        all_defs = loader.get_all()
        assert len(all_defs) >= 3  # At least temp, field, position
        assert all(isinstance(d, ParameterDefinition) for d in all_defs)


class TestParameterState:
    """Test ParameterState NamedTuple."""

    def test_parameter_state_creation(self):
        """Test creating ParameterState."""
        state = ParameterState(
            value=4.2,
            is_swept=False,
            min=None,
            max=None,
            sweep_direction=SweepDirection.UNDEFINED,
            unit='K'
        )
        assert state.value == 4.2
        assert state.is_swept is False
        assert state.unit == 'K'

    def test_parameter_state_range_property(self):
        """Test range property."""
        # Swept parameter
        state_swept = ParameterState(
            value=None,
            is_swept=True,
            min=-5.0,
            max=5.0,
            sweep_direction=SweepDirection.INCREASING,
            unit='T'
        )
        assert state_swept.range == (-5.0, 5.0)

        # Fixed parameter
        state_fixed = ParameterState(
            value=4.2,
            is_swept=False,
            min=None,
            max=None,
            sweep_direction=SweepDirection.UNDEFINED,
            unit='K'
        )
        assert state_fixed.range is None

    def test_parameter_state_sweep_property(self):
        """Test sweep property."""
        state = ParameterState(
            value=None,
            is_swept=True,
            min=-5.0,
            max=5.0,
            sweep_direction=SweepDirection.INCREASING,
            unit='T'
        )
        assert state.sweep == (-5.0, 5.0, SweepDirection.INCREASING)


class TestParameterIntegration:
    """Integration tests for Parameter parsing."""

    def test_parse_temperature_examples(self):
        """Test parsing realistic temperature parameter strings."""
        loader = DefinitionsLoader()
        temp_def = loader.get_definition('temperature')

        # Fixed temperature
        param1 = Parameter(temp_def)
        assert param1.parse_fixed("T=4.2K") is True
        assert param1.state.value == Decimal('4.2')

        # Sweep temperature
        param2 = Parameter(temp_def)
        assert param2.parse_sweep("sweepTemperature") is True
        assert param2.state.is_swept is True

    def test_parse_field_examples(self):
        """Test parsing realistic field parameter strings."""
        loader = DefinitionsLoader()
        field_def = loader.get_definition('magnetic_field')

        # Fixed field
        param1 = Parameter(field_def)
        assert param1.parse_fixed("B=0T") is True
        assert param1.state.value == Decimal('0')

        # Range field
        param2 = Parameter(field_def)
        assert param2.parse_range("B-14to14T") is True
        assert param2.state.min_val == Decimal('-14')
        assert param2.state.max_val == Decimal('14')

    def test_parse_position_with_special_values(self):
        """Test parsing position parameter with special values."""
        loader = DefinitionsLoader()
        pos_def = loader.get_definition('position')

        # Parse using special value
        param = Parameter(pos_def)
        param.set_fixed("OOP")
        assert param.state.value == Decimal('0')


class TestPrefixParsing:
    """Test automatic SI prefix parsing functionality."""

    def test_parse_millikelvin(self):
        """Test parsing temperature with milli prefix."""
        temp_def = ParameterDefinition(
            name_id='temperature',
            long_names=['temperature'],
            short_names=['T'],
            units=['K']  # Only base unit 'K', not 'mK'
        )
        param = Parameter(temp_def)
        assert param.parse_fixed('T=100mK') is True
        assert param.state.value == Decimal('0.1')
        assert param.state.unit == 'K'

    def test_parse_microkelvin(self):
        """Test parsing temperature with micro prefix."""
        temp_def = ParameterDefinition(
            name_id='temperature',
            long_names=['temperature'],
            short_names=['T'],
            units=['K']
        )
        param = Parameter(temp_def)
        assert param.parse_fixed('T=500uK') is True
        assert param.state.value == Decimal('0.0005')
        assert param.state.unit == 'K'

    def test_parse_millitesla_range(self):
        """Test parsing field range with milli prefix - cleaner syntax."""
        field_def = ParameterDefinition(
            name_id='field',
            long_names=['field'],
            short_names=['B'],
            units=['T']  # Only base unit 'T', not 'mT'
        )
        param = Parameter(field_def)
        # Note: prefix specified once (5mT), not twice (5mto5mT)
        assert param.parse_range('B-5to5mT') is True
        assert param.state.min_val == Decimal('-0.005')
        assert param.state.max_val == Decimal('0.005')
        assert param.state.unit == 'T'

    def test_parse_without_prefix_still_works(self):
        """Test that unprefixed units still work correctly."""
        field_def = ParameterDefinition(
            name_id='field',
            long_names=['field'],
            short_names=['B'],
            units=['T']
        )
        param = Parameter(field_def)
        assert param.parse_range('B-14to14T') is True
        assert param.state.min_val == Decimal('-14')
        assert param.state.max_val == Decimal('14')
        assert param.state.unit == 'T'

    def test_various_si_prefixes(self):
        """Test various SI prefixes are recognized."""
        volt_def = ParameterDefinition(
            name_id='voltage',
            long_names=['voltage'],
            short_names=['V'],
            units=['V']
        )

        # kilo
        param1 = Parameter(volt_def)
        assert param1.parse_fixed('V=1kV') is True
        assert param1.state.value == Decimal('1000')

        # nano
        param2 = Parameter(volt_def)
        assert param2.parse_fixed('V=500nV') is True
        assert param2.state.value == Decimal('5E-7')

        # mega
        param3 = Parameter(volt_def)
        assert param3.parse_fixed('V=2MV') is True
        assert param3.state.value == Decimal('2000000')

    def test_prefix_not_in_base_units_ignored(self):
        """Test that prefixes are only applied to defined base units."""
        # If we define 'mT' as a base unit, 'mmT' should not be interpreted as milli-mT
        field_def = ParameterDefinition(
            name_id='field',
            long_names=['field'],
            short_names=['B'],
            units=['T', 'mT']  # Both T and mT as base units
        )
        param = Parameter(field_def)
        assert param.parse_fixed('B=5mT') is True
        # Should recognize 'mT' as prefixed 'T'
        assert param.state.value == Decimal('0.005')
        assert param.state.unit == 'T'
