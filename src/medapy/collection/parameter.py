from dataclasses import dataclass, field
from typing import Dict, Pattern, Iterable, NamedTuple
from types import MappingProxyType
import re
from enum import Enum
from pathlib import Path
from decimal import Decimal
import json

from medapy.utils.prefixes import SYMBOL_TO_MULTIPLIER, format_with_prefix, detect_min_precision


class SweepDirection(Enum):
    """
    Direction of a parameter sweep.

    Lookup accepts the value or a case-insensitive alias: ``'inc'``/``'up'``/
    ``'increasing'``, ``'dec'``/``'down'``/``'decreasing'`` and
    ``'und'``/``'na'``/``'undefined'``. Members also compare equal to these
    strings.

    Attributes
    ----------
    INCREASING : int
        Sweep from lower to higher values, ``1``.
    DECREASING : int
        Sweep from higher to lower values, ``-1``.
    UNDEFINED : int
        Direction unknown, ``0``.

    Examples
    --------
    >>> SweepDirection('up')
    <SweepDirection.INCREASING: 1>
    >>> SweepDirection.DECREASING == 'dec'
    True
    """
    INCREASING = 1
    DECREASING = -1
    UNDEFINED = 0

    __aliases_increasing = frozenset(('inc', 'up', 'increasing'))
    __aliases_decreasing = frozenset(('dec', 'down', 'decreasing'))
    __aliases_undefined = frozenset(('und', 'na', 'undefined'))

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in cls.__aliases_increasing:
                return cls.INCREASING
            elif value in cls.__aliases_decreasing:
                return cls.DECREASING
            elif value in cls.__aliases_undefined:
                return cls.UNDEFINED

    def __eq__(self, other):
        if isinstance(other, str):
            other = self._missing_(other)
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


@dataclass
class ParameterDefinition:
    """
    Schema of one parameter: its names, units and the patterns that parse it.

    Collection classes take a list of definitions and parse every filename
    against each of them. Usually built by `DefinitionsLoader` from JSON rather
    than directly.

    Parameters
    ----------
    name_id : str
        Identifier used to look the parameter up, e.g. ``'temperature'``.
    long_names : str or iterable of str
        Full names, e.g. ``'Field'``. Generated filenames use a long name for
        swept parameters (``sweepField``).
    short_names : str or iterable of str, optional
        Abbreviations, e.g. ``'T'``. Generated filenames use a short name for
        fixed parameters (``T=10K``). Parsing accepts long and short names alike.
        Must not overlap `long_names`; at least one of the two must be non-empty.
    units : str or iterable of str, optional
        Accepted units. SI-prefixed forms (``mK``, ``uT``) are accepted too.
    special_values : mapping of str to number, optional
        Named values, e.g. ``{'IP': 90, 'OOP': 0}``.
    patterns : mapping of str to str, optional
        Regex templates overriding or adding to the defaults ``'fixed'``,
        ``'range'`` and ``'sweep'``. Templates may use the placeholders
        ``{NAME}``, ``{LNAME}``, ``{SNAME}``, ``{VALUE}`` and ``{UNIT}``.
    si_formatting : bool, default True
        Whether to write values with SI prefixes when generating filenames.
    precision : int or 'auto', default 'auto'
        Decimal places when writing values; 'auto' drops trailing zeros.

    Raises
    ------
    ValueError
        If both name sets are empty, they overlap, or a pattern is invalid.

    See Also
    --------
    DefinitionsLoader : Load definitions from JSON
    """
    name_id: str
    long_names: str | Iterable[str]
    short_names: str | Iterable[str] = frozenset()
    units: str | Iterable[str] = frozenset()
    special_values: MappingProxyType[str, Decimal] = field(
        default_factory=lambda: MappingProxyType({}))
    patterns: MappingProxyType[str, Pattern] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False)
    si_formatting: bool = True  # Whether to format with SI prefixes
    precision: int | str = 'auto'  # Precision for value formatting ('auto' or int)

    def __post_init__(self):
        self._to_immutables()
        self._validate_names()
        self._compile_patterns()

    def _to_immutables(self):
        """Convert to immutable types"""
        self.long_names = frozenset(self.long_names)
        self.short_names = frozenset(self.short_names)
        self.units = frozenset(self.units)
        self.special_values = MappingProxyType(
            {key: Decimal(str(val)) for key, val in (self.special_values or {}).items()})
        if not isinstance(self.patterns, MappingProxyType):
            self.patterns = MappingProxyType(self.patterns or {})

    def _validate_names(self) -> None:
        if not (self.long_names or self.short_names):
            raise ValueError("Long and short names cannot be empty simultaneously")
        overlap = self.long_names & self.short_names
        if overlap:
            raise ValueError(f"Long and short names overlap: {overlap}")

    def _create_base_patterns(self) -> Dict[str, str]:
        special_values_pattern = '|'.join(map(re.escape, self.special_values.keys()))
        # Only add special values alternative if there are any
        if special_values_pattern:
            value_pattern = rf'-?\d+\.?\d*(?:[eE][+-]?\d+)?|{special_values_pattern}'
        else:
            value_pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'

        # Generate prefixed units (e.g., 'K' -> 'K', 'mK', 'uK', 'nK', etc.)
        prefixed_units = set(self.units)
        for unit in self.units:
            for prefix_symbol in SYMBOL_TO_MULTIPLIER:
                prefixed_units.add(f'{prefix_symbol}{unit}')

        return {
            'LNAME': '|'.join(map(re.escape, self.long_names)),
            'SNAME': '|'.join(map(re.escape, self.short_names)),
            'NAME': '|'.join(map(re.escape, self.long_names | self.short_names)),
            'UNIT': '|'.join(map(re.escape, prefixed_units)),
            'VALUE': value_pattern,
        }

    def _compile_patterns(self) -> None:
        base = self._create_base_patterns()

        # Default patterns
        default_patterns = {
            'sweep': r'sweep{NAME}|{NAME}sweep',
            'range': r'{NAME}{VALUE}to{VALUE}{UNIT}?$',
            'fixed': r'{NAME}=?{VALUE}{UNIT}?$',
        }

        # Merge with custom patterns if provided
        patterns_to_compile = default_patterns
        if self.patterns:
            patterns_to_compile.update(self.patterns)

        # Replace placeholders and compile
        compiled_patterns = dict()
        for pattern_name, pattern_template in patterns_to_compile.items():
            # Replace all placeholders with actual patterns
            final_pattern = pattern_template
            for placeholder, value in base.items():
                if placeholder == 'VALUE' or placeholder == 'UNIT':
                    final_pattern = final_pattern.replace(
                        f'{{{placeholder}}}', f'({value})')
                else:
                    final_pattern = final_pattern.replace(
                        f'{{{placeholder}}}', f'(?:{value})')

            try:
                compiled_patterns[pattern_name] = re.compile(final_pattern)
            except re.error as e:
                raise ValueError(f"Invalid pattern '{pattern_name}': {e}")
        self.patterns = MappingProxyType(compiled_patterns)

    def get_pattern(self, pattern_type: str) -> Pattern:
        """Return the compiled pattern of a type ('fixed', 'range', 'sweep', ...).

        Raises
        ------
        ValueError
            If no pattern of that type is defined.
        """
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        return self.patterns[pattern_type]

    def match(self, pattern_type: str, text: str) -> re.Match | None:
        """Match the pattern of `pattern_type` at the start of `text`."""
        return self.get_pattern(pattern_type).match(text)

    def search(self, pattern_type: str, text: str) -> re.Match | None:
        """Search for the pattern of `pattern_type` anywhere in `text`."""
        return self.get_pattern(pattern_type).search(text)


@dataclass
class _ParameterState:
    """Mutable instance-specific state"""
    value: Decimal | None = None
    is_swept: bool = False
    min_val: Decimal | None = None
    max_val: Decimal | None = None
    sweep_direction: SweepDirection = SweepDirection.UNDEFINED
    unit: str | None = None

    def __eq__(self, other):
        for attr, value in self.__dict__.items():
            if value != other.__dict__.get(attr):
                return False
        return True

    def __copy__(self):
        # Method is not tested
        return type(self)(**self.__dict__.copy())


class Parameter:
    """
    A parameter of one file: a shared `ParameterDefinition` plus its own state.

    The state is either fixed (a single value) or swept (optionally with a range
    and a direction). Values are stored as `Decimal`; read them through
    `MeasurementFile.state_of`, which returns a `ParameterState`.

    Parameters
    ----------
    definition : ParameterDefinition
        Schema of the parameter.
    **kwargs
        Initial state: ``value``, ``is_swept``, ``min_val``, ``max_val``,
        ``sweep_direction``, ``unit``. A given ``value`` makes the parameter
        fixed; given ``min_val`` and ``max_val`` make it swept.

    Examples
    --------
    >>> temp_def = DefinitionsLoader().get_definition('temperature')
    >>> p = Parameter(temp_def)
    >>> p.parse_fixed('T=1.5K')
    True
    >>> str(p)
    'T=1.5K'
    """
    def __init__(self, definition: ParameterDefinition, **kwargs):
        self.definition = definition
        self.state = _ParameterState(**kwargs)
        self.check_state()

    def check_state(self):
        """Normalize the state: apply `set_fixed`/`set_swept` from the stored values."""
        if self.state.value is not None:
            self.set_fixed(self.state.value)

        if self.state.min_val is not None and self.state.max_val is not None:
            self.set_swept(self.state.min_val, self.state.max_val)

    def _decompose_prefixed_unit(self, unit_str: str | None) -> tuple[float, str | None]:
        """
        Decompose a potentially prefixed unit into multiplier and base unit.

        Parameters
        ----------
        unit_str : str or None
            Unit string that may include an SI prefix (e.g., 'mK', 'uT', 'K')

        Returns
        -------
        tuple[float, str | None]
            (multiplier, base_unit) where multiplier is 1.0 for unprefixed units

        Examples
        --------
        >>> _decompose_prefixed_unit('mK')
        (0.001, 'K')
        >>> _decompose_prefixed_unit('K')
        (1.0, 'K')
        >>> _decompose_prefixed_unit(None)
        (1.0, None)
        """
        if not unit_str:
            return 1.0, unit_str

        # Check if unit starts with a known SI prefix
        if len(unit_str) > 1 and unit_str[0] in SYMBOL_TO_MULTIPLIER:
            prefix = unit_str[0]
            base_unit = unit_str[1:]
            # Verify the base unit is in the defined units
            if base_unit in self.definition.units:
                multiplier = SYMBOL_TO_MULTIPLIER[prefix]
                return multiplier, base_unit

        # No prefix found or base unit not in definitions
        return 1.0, unit_str

    def parse_fixed(self, text: str) -> bool:
        """Set a fixed value parsed from `text` (e.g. ``'T=10K'``).

        Returns
        -------
        bool
            True if `text` matched the definition's 'fixed' pattern.
        """
        m = self.definition.match('fixed', text)
        if not m:
            return False
        value_str = m.group(1)
        unit_str = m.group(2)

        # Decompose unit into prefix multiplier and base unit
        multiplier, base_unit = self._decompose_prefixed_unit(unit_str)

        # Parse value and apply prefix multiplier
        value = self.decimal_of(value_str)
        scaled_value = value * Decimal(str(multiplier))

        self.set_fixed(scaled_value)
        self.state.unit = base_unit
        return True

    def parse_range(self, text: str) -> bool:
        """Set a swept range parsed from `text` (e.g. ``'B-14to14T'``).

        Returns
        -------
        bool
            True if `text` contains the definition's 'range' pattern.
        """
        m = self.definition.search('range', text)
        if not m:
            return False
        start_str, end_str = m.group(1), m.group(2)
        unit_str = m.group(3)

        # Decompose unit into prefix multiplier and base unit
        multiplier, base_unit = self._decompose_prefixed_unit(unit_str)

        # Parse values and apply prefix multiplier
        start_val = self.decimal_of(start_str) * Decimal(str(multiplier))
        end_val = self.decimal_of(end_str) * Decimal(str(multiplier))

        self.set_swept(start_val, end_val)
        self.state.unit = base_unit
        return True

    def parse_sweep(self, text: str) -> bool:
        """Mark the parameter as swept if `text` is a range or sweep token.

        A range (``'B-14to14T'``) also sets the bounds and direction; a bare
        sweep token (``'sweepField'``) only sets the swept flag.

        Returns
        -------
        bool
            True if `text` matched either pattern.
        """
        is_range = self.parse_range(text)
        if is_range:
            return True

        m = self.definition.match('sweep', text)
        if not m:
            return False
        self.state.is_swept = True
        return True

    def set_fixed(self, value: float | str):
        """Make the parameter fixed at `value`, clearing any sweep state.

        `value` may be a number or a special value name (e.g. ``'IP'``).
        Returns the parameter itself.
        """
        self.state.is_swept = False
        self.state.value = self.decimal_of(value)
        # Reset sweep-related attributes
        self.state.min_val = None
        self.state.max_val = None
        self.state.sweep_direction = SweepDirection.UNDEFINED
        return self

    def set_swept(self, start_val: Decimal, end_val: Decimal):
        """Make the parameter swept from `start_val` to `end_val`.

        The bounds are stored ordered as min/max and the direction is derived
        from their order. If either is None, the sweep has no range and an
        undefined direction. Returns the parameter itself.
        """
        self.state.is_swept = True
        # Reset fixed-related attributes
        self.state.value = None

        if start_val is not None and end_val is not None:
            start_val, end_val = self.decimal_of(start_val), self.decimal_of(end_val)
            if end_val > start_val:
                self.state.sweep_direction = SweepDirection.INCREASING
            elif end_val < start_val:
                start_val, end_val = end_val, start_val
                self.state.sweep_direction = SweepDirection.DECREASING
            self.state.min_val = start_val
            self.state.max_val = end_val
        else:
            # Undefined sweep - reset sweep values and direction
            self.state.min_val = None
            self.state.max_val = None
            self.state.sweep_direction = SweepDirection.UNDEFINED
        return self

    def update(self, other):
        """Overwrite state fields with the truthy fields of another parameter."""
        self.state.value = other.state.value or self.state.value
        self.state.is_swept = other.state.is_swept or self.state.is_swept
        self.state.min_val = other.state.min_val or self.state.min_val
        self.state.max_val = other.state.max_val or self.state.max_val
        self.state.sweep_direction = (
            other.state.sweep_direction or self.state.sweep_direction
        )
        self.state.unit = other.state.unit or self.state.unit

    def decimal_of(self, value_str):
        """Convert string value to Decimal, handling special values"""
        if not isinstance(value_str, str):
            value_str = str(value_str)
        if value_str in self.definition.special_values:
            return self.definition.special_values[value_str]
        return Decimal(value_str)

    def copy(self):
        """Return a copy with its own state and the same definition."""
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.definition, **self.state.__dict__.copy())

    def __str__(self):
        if self.definition.long_names:
            lname = sorted(list(self.definition.long_names))[0]
        else:
            lname = None
        if self.definition.short_names:
            sname = sorted(list(self.definition.short_names))[0]
        else:
            sname = None

        unit = self.state.unit or ""

        if self.state.is_swept:
            name = lname or sname
            s = f'sweep{name}'
            if isinstance(self.state.min_val, Decimal) and isinstance(
                self.state.max_val, Decimal
            ):
                mn, mx = self.state.min_val, self.state.max_val
                if self.state.sweep_direction == SweepDirection.DECREASING:
                    mx, mn = mn, mx

                # Format min and max values
                if self.definition.si_formatting:
                    mn_str = format_with_prefix(
                        float(mn),
                        precision=self.definition.precision,
                        use_space=False
                    )
                    mx_str = format_with_prefix(
                        float(mx),
                        precision=self.definition.precision,
                        use_space=False
                    )
                else:
                    # No SI prefix formatting
                    if self.definition.precision == 'auto':
                        prec_mn = detect_min_precision(float(mn))
                        prec_mx = detect_min_precision(float(mx))
                        mn_str = f'{float(mn):.{prec_mn}f}'
                        mx_str = f'{float(mx):.{prec_mx}f}'
                    else:
                        prec = self.definition.precision
                        mn_str = f'{float(mn):.{prec}f}'
                        mx_str = f'{float(mx):.{prec}f}'

                s += f'{mn_str}to{mx_str}{unit}'
            return s
        else:
            name = sname or lname
            value = float(self.state.value)

            # Format value with or without SI prefixes
            if self.definition.si_formatting:
                value_str = format_with_prefix(
                    value,
                    precision=self.definition.precision,
                    use_space=False
                )
            else:
                # No SI prefix formatting
                if self.definition.precision == 'auto':
                    prec = detect_min_precision(value)
                    value_str = f'{value:.{prec}f}'
                else:
                    prec = self.definition.precision
                    value_str = f'{value:.{prec}f}'

            return f'{name}={value_str}{unit}'

    def __repr__(self):
        return ('Parameter: '
                f'[type: {self.definition.name_id}, '
                f'value: {self.state.value}, is_swept: {self.state.is_swept}, '
                f'min_val: {self.state.min_val}, max_val: {self.state.max_val}, '
                f'sweep_direction: {self.state.sweep_direction}]')

class DefinitionsLoader:
    """
    Registry of parameter definitions loaded from JSON files.

    The built-in definitions (``magnetic_field``, ``temperature``, ``position``)
    are always loaded first; a custom file adds to them and replaces entries
    with the same name.

    Parameters
    ----------
    custom_config_path : str or Path, optional
        JSON file with additional definitions.

    Examples
    --------
    >>> loader = DefinitionsLoader('my_params.json')
    >>> collection = MeasurementCollection(folder, parameters=loader.get_all())
    """
    def __init__(self, custom_config_path: Path | str = None):
        self._definitions = {}
        self._load_defaults()
        if custom_config_path:
            self.load_definitions(custom_config_path)

    def _load_defaults(self):
        """Load built-in parameter definitions"""
        default_path = Path(__file__).parent / 'parameter_definitions.json'
        self.load_definitions(default_path)

    def load_definitions(self, path: Path | str):
        """Load custom parameter definitions from JSON"""
        with open(path, 'r') as f:
            definitions = json.load(f)

        for name, definition in definitions.items():
            # Convert lists to frozensets and MappingProxyType
            definition['name_id'] = name
            definition['long_names'] = frozenset(definition.get('long_names', []))
            definition['short_names'] = frozenset(definition.get('short_names', []))
            definition['units'] = frozenset(definition.get('units', []))
            definition['special_values'] = MappingProxyType(definition.get('special_values', {}))
            definition['patterns'] = MappingProxyType(definition.get('patterns', {}))
            # New formatting options with defaults
            definition['si_formatting'] = definition.get('si_formatting', True)
            definition['precision'] = definition.get('precision', 'auto')
            self._definitions[name] = definition

    def get_definition_dict(self, name: str) -> dict[str, set]:
        """Get parameter definition by name"""
        if name not in self._definitions:
            raise KeyError(f"Parameter definition '{name}' not found")
        return self._definitions[name]

    def get_definition(self, name: str) -> ParameterDefinition:
        """Create a `ParameterDefinition` from the definition named `name`."""
        definition = self.get_definition_dict(name)
        return ParameterDefinition(**definition)

    def get_all(self) -> list[ParameterDefinition]:
        """Create a `ParameterDefinition` for every loaded definition."""
        return [self.get_definition(name) for name in self._definitions.keys()]


class ParameterState(NamedTuple):
    """
    Read-only snapshot of a parameter's state, with numbers as floats.

    Returned by `MeasurementFile.state_of`.

    Attributes
    ----------
    value : float or None
        Value of a fixed parameter; None if swept.
    is_swept : bool
        Whether the parameter is swept.
    min, max : float or None
        Bounds of a swept range; None if fixed or the range is unknown.
    sweep_direction : SweepDirection
        Direction of the sweep.
    unit : str or None
        Unit as written in the filename, without SI prefix.
    """
    value: float | None
    is_swept: bool
    min: float | None
    max: float | None
    sweep_direction: SweepDirection
    unit: str | None

    @staticmethod
    def to_float(val) -> float | None:
        """Convert to float, passing None through."""
        return float(val) if val is not None else None

    @classmethod
    def from_state(cls, state: _ParameterState):
        """Build a snapshot from a `Parameter`'s internal state."""
        return cls(
            value=cls.to_float(state.value),
            is_swept=state.is_swept,
            min=cls.to_float(state.min_val),
            max=cls.to_float(state.max_val),
            sweep_direction=state.sweep_direction,
            unit=state.unit,
        )

    @property
    def range(self) -> tuple[float, float] | None:
        """``(start, end)`` in sweep order, or None if not a swept range."""
        if self.is_swept and self.min is not None and self.max is not None:
            # Return (start, end) preserving sweep direction
            if self.sweep_direction == SweepDirection.DECREASING:
                return (self.max, self.min)
            return (self.min, self.max)
        return None

    @property
    def sweep(self) -> tuple[float, float, SweepDirection] | None:
        """``(min, max, direction)`` if swept, else None."""
        if self.is_swept:
            return (self.min, self.max, self.sweep_direction)
        return None
