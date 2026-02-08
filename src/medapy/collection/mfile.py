from __future__ import annotations
from dataclasses import dataclass
from itertools import zip_longest
import re
import warnings
from pathlib import Path
from typing import Iterable

from .contact import ContactPair
from .parameter import ParameterDefinition, DefinitionsLoader, Parameter, ParameterState, SweepDirection
from medapy.utils import validations


@dataclass(frozen=False)
class MeasurementFile:
    """
    Represents a measurement file with automatic parameter extraction from filename.

    Parses filenames to extract measurement parameters (temperature, field, etc.) and
    contact pair configurations. Provides filtering and state access methods.

    Parameters
    ----------
    path : str or Path
        Path to the measurement file.
    parameters : list of ParameterDefinition or Parameter, or Path or str
        Either a list of parameter definition/instances, or path to parameter
        definitions file (JSON).
    separator : str, default "_"
        Character used to separate filename parts.

    Attributes
    ----------
    path : Path
        Path to the measurement file.
    parameters : dict of {str : Parameter}
        Parsed parameters from filename, keyed by parameter name_id.
    contact_pairs : list of ContactPair
        Contact configurations extracted from filename.
    separator : str
        Filename parts separator.

    Examples
    --------
    >>> from medapy.collection import MeasurementFile, DefinitionsLoader
    >>> # Create file with parameter parsing
    >>> parameters = DefinitionsLoader().get_all()
    >>> mfile = MeasurementFile(
    ...     "sample_V1-5(1mV)_B-14to14T_T=3K.csv",
    ...     parameters=parameters
    ... )
    >>> # Access parsed values
    >>> temp = mfile.value_of('temperature')  # Returns 3.0
    >>> field_range = mfile.range_of('magnetic_field')  # Returns (-14.0, 14.0)
    >>> # Check file properties
    >>> has_voltage = mfile.check(polarization='V')
    >>> low_temp = mfile.check(temperature=(0, 5))

    See Also
    --------
    MeasurementCollection : Manage collections of measurement files
    ParameterDefinition : Define parameter schemas
    ContactPair : Represent contact pair configurations

    Notes
    -----
    The filename is automatically parsed on initialization to extract parameters
    and contact pairs based on the provided parameter definitions.
    """
    path: Path
    parameters: dict[str, Parameter]
    contact_pairs: list[ContactPair]
    separator: str = "_"

    def __init__(
        self,
        path: str | Path,
        parameters: list[ParameterDefinition | Parameter] | Path | str,
        separator: str = "_",
    ):
        """
        Initialize MeasurementFile with automatic filename parsing.

        Parameters
        ----------
        path : str or Path
            Path to the measurement file.
        parameters : list of ParameterDefinition or Parameter, or Path or str
            Parameter definitions for parsing. Can be:
            - List of ParameterDefinition objects
            - List of Parameter instances (copies parameters directly)
            - Path to parameter definitions JSON file
            - String path to parameter definitions JSON file
        separator : str, default "_"
            Character used to separate parts in the filename.

        Examples
        --------
        >>> from medapy.collection import MeasurementFile, DefinitionsLoader
        >>> # Using default parameter definitions
        >>> params = DefinitionsLoader().get_all()
        >>> mfile = MeasurementFile("data_T=4K.csv", parameters=params)
        >>> # Using custom parameter definitions file
        >>> mfile = MeasurementFile("data_T=4K.csv", parameters="custom_params.json")
        """
        self.path = Path(path)
        self.separator = separator
        self.contact_pairs = []

        # Initialize parameters dictionary
        if isinstance(parameters, (str, Path)):
            param_defs = DefinitionsLoader(parameters)
            self.param_definitions = {dfn.name_id: dfn for dfn in param_defs.get_all()}
        else:
            item_type = type(parameters[0])
            validations.class_in_iterable(parameters, item_type, iter_name='parameters')
            # Convert list of parameters to dictionary
            self.param_definitions = dict()
            if item_type == Parameter:
                self.parameters = dict()
                for param in parameters:
                    name = param.definition.name_id
                    self.param_definitions[name] = param.definition
                    self.parameters[name] = param.copy()
                return
            if item_type == ParameterDefinition:
                for param in parameters:
                    name = param.name_id
                    self.param_definitions[name] = param

        self.parameters = dict()
        self._parse_filename()

    @property
    def name(self) -> str:
        return self.path.name

    def check(
        self,
        *,
        mode: str = 'all',
        contacts: tuple[int, int] | list[tuple[int, int] | int | str | ContactPair] | int | str | ContactPair | None = None,
        polarization: str | None = None,
        sweeps: list[str] | str | None = None,
        sweep_directions: list[str | None] | str | None = None,
        exact_sweep: bool = True,
        name_contains: list[str] | str | None = None,
        **parameter_filters: dict,
    ) -> bool:
        """
        Check if file matches filter conditions with configurable logic.

        Parameters
        ----------
        mode : {'all', 'any'}, default 'all'
            Logic mode: 'all' requires all criteria to match (AND logic),
            'any' requires at least one criterion to match (OR logic).
        contacts : various types, optional
            Contact specification to match:
            - int: single contact (e.g., 1)
            - tuple: contact pair (e.g., (1, 2), (1, 2, 'I'), or (1, 2, 'I', 1e-6))
            - str: string representation (e.g., "I1-2(2uA)")
            - ContactPair: full contact pair object
            - list: multiple contact specifications
        polarization : {'I', 'V'}, optional
            'I' for current polarization or 'V' for voltage polarization.
        sweeps : str or list of str, optional
            Parameter name(s) that should be swept.
        sweep_directions : {'inc', 'dec'} or list, optional
            Expected sweep direction(s): 'inc' for increasing, 'dec' for decreasing.
        exact_sweep : bool, default True
            If True, sweep ranges must match exactly. If False, file's sweep
            range must be contained within the specified range.
        name_contains : str or list of str, optional
            String(s) or regex pattern(s) that must appear in filename.
        **parameter_filters : keyword arguments
            Filter by parameter values. Use parameter name as key with:
            - Single value: checks for exact match (fixed parameters)
            - Tuple (min, max): checks if value is in range
            - Suffix '_sweep': checks for swept parameters with the given range

        Returns
        -------
        bool
            True if file matches criteria according to the specified mode.

        Examples
        --------
        >>> # Check for voltage polarization at low temperature
        >>> mfile.check(polarization='V', temperature=(0, 5))
        >>> # Check for field sweep with increasing direction
        >>> mfile.check(sweeps='magnetic_field', sweep_directions='inc')
        >>> # Check for specific contacts
        >>> mfile.check(contacts=(1, 5))
        >>> mfile.check(contacts='I1-5(10mA)')
        >>> # Use OR logic (any criterion matches)
        >>> mfile.check(mode='any', temperature=4.2, magnetic_field=0)
        >>> # Check filename contains pattern
        >>> mfile.check(name_contains='sample.*_run1')

        See Also
        --------
        check_contacts : Check only contact configuration
        check_sweep : Check only sweep parameters
        check_parameter : Check individual parameter
        MeasurementCollection.filter : Filter collection using check criteria
        """

        def _check_generator():
            """Lazy generator of individual check results"""
            if contacts is not None:
                yield self.check_contacts(contacts)

            if polarization is not None:
                yield self.check_polarization(polarization)

            if sweeps is not None:
                yield self.check_sweeps(sweeps, sweep_directions)

            if name_contains is not None:
                yield self.check_name(name_contains)

            # Check parameter filters
            for param_name, filter_value in parameter_filters.items():
                if param_name.endswith('_sweep'):
                    # Handle sweep parameter filtering
                    base_name = param_name[:-6]  # Remove '_sweep' suffix
                    yield self.check_parameter(
                        base_name, filter_value, swept=True, exact_sweep=exact_sweep
                    )
                else:
                    # Handle fixed parameter filtering
                    yield self.check_parameter(
                        param_name, filter_value, swept=False, exact_sweep=exact_sweep
                    )

        if mode == 'all':
            return all(_check_generator())
        elif mode == 'any':
            return any(_check_generator())
        else:
            raise ValueError(f"mode must be 'all' or 'any', got '{mode}'")

    def check_polarization(self, polarization: str):
        return any(pair.polarization == polarization for pair in self.contact_pairs)

    def check_name(self, strings: list[str] | str) -> bool:
        """Check if filename contains specified string(s) or matches regex pattern(s)"""
        if not isinstance(strings, (list, tuple)):
            strings = [strings]

        filename = self.name

        for pattern in strings:
            try:
                if re.search(pattern, filename):
                    continue
                else:
                    return False
            except re.error as e:
                warnings.warn(
                    f"Invalid regex pattern '{pattern}': {e}. "
                    f"Falling back to substring matching.",
                    UserWarning,
                )
                if pattern in filename:
                    continue
                else:
                    return False

        return True

    def check_sweeps(
        self, sweeps: list[str] | str, directions: list[str | None] | str | None = None
    ):
        if not isinstance(sweeps, (list, tuple)):
            sweeps = [sweeps]
        if not isinstance(directions, (list, tuple)):
            directions = [directions]
        if len(sweeps) < len(directions):
            raise ValueError(
                f"Number of sweeps ({len(sweeps)}) is smaller "
                f"than number of directions ({len(directions)})"
            )
        sweeps_and_dirs = zip_longest(sweeps, directions)

        return all(self.check_sweep(*sweep) for sweep in sweeps_and_dirs)

    def check_sweep(self, sweep: str | None, direction: str | None = None):
        param = self.parameters.get(sweep)
        if not param:
            return False

        is_swept = param.state.is_swept
        if not direction:
            return is_swept

        is_correct_direction = param.state.sweep_direction == direction
        return is_swept and is_correct_direction

    def check_contacts(
        self, contacts: tuple[int, int] | list[tuple[int, int] | int | str | ContactPair] | int | str | ContactPair
    ) -> bool:
        """Check if file contains specified contact configuration"""

        # Convert single pair/contact to list
        if not isinstance(contacts, list):
            contacts = [contacts]

        # Check if all specified contacts/pairs are present
        return all(
            any(pair.pair_matches(check_pair) for pair in self.contact_pairs)
            for check_pair in contacts
        )

    def check_parameter(
        self,
        name: str,
        value: list[float, float] | float,
        swept: bool | None = None,
        exact_sweep: bool = True,
    ) -> bool:
        """Check if parameter matches value or range"""
        param = self.parameters.get(name)
        if not param:
            return False

        if swept is not None:
            # Filter by swept state if specified
            if swept and not param.state.is_swept:
                return False
            if not swept and param.state.is_swept:
                return False

        # Delegate to appropriate implementation
        if param.state.is_swept:
            return self._check_sweep_parameter(name, value, exact_sweep)
        else:
            return self._check_fixed_parameter(name, value)

    def _validate_parameter_range(self, name: str, value) -> tuple:
        """
        Validate and prepare parameter range for filtering.

        Handles:
        - Numeric ranges: (5, 10)
        - Open boundaries: (5, None) or (None, 10)
        - Special string values: ('IP', 'OOP') for parameters with shortcuts

        Args:
            name: Parameter name
            value: Range tuple/list (min, max) where values can be numeric, string, or None

        Returns:
            tuple: (min_val, max_val) as Decimal, ready for comparison

        Raises:
            TypeError: If value is a string or not iterable
            ValueError: If value doesn't have exactly 2 elements
        """
        param = self.parameters.get(name)

        # Strings are iterable but shouldn't be treated as ranges
        if isinstance(value, str):
            raise TypeError(
                f"Range for parameter '{name}' cannot be a string. Got: '{value}'"
            )

        try:
            range_list = list(value)
        except TypeError:
            raise TypeError(
                f"Range for parameter '{name}' must be an iterable (tuple, list, etc.)"
            )

        if len(range_list) != 2:
            raise ValueError(
                f"Range for parameter '{name}' must contain exactly 2 values, "
                f"got {len(range_list)}"
            )

        min_val, max_val = range_list

        # Convert None to infinity, let param.decimal_of handle everything else
        if min_val is None:
            min_val = param.decimal_of('-inf')
        else:
            # Handle both numbers and special string values
            min_val = param.decimal_of(str(min_val))

        if max_val is None:
            max_val = param.decimal_of('inf')
        else:
            max_val = param.decimal_of(str(max_val))

        # Swap if needed
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        return min_val, max_val

    def _check_fixed_parameter(
        self, name: str, value: list[float, float] | float
    ) -> bool:
        """Protected: Handle fixed parameter logic"""
        param = self.parameters.get(name)

        # Handle exact value (including string special values like 'OOP', 'IP')
        # Strings are iterable, but are single values
        if isinstance(value, str) or not isinstance(value, Iterable):
            return param.state.value == param.decimal_of(str(value))

        # Handle range - for fixed parameter, check if value is within range
        min_val, max_val = self._validate_parameter_range(name, value)
        return min_val <= param.state.value <= max_val

    def _check_sweep_parameter(
        self, name: str, value: list[float, float] | float, exact_sweep: bool = True
    ) -> bool:
        """Protected: Handle sweep parameter logic"""
        param = self.parameters.get(name)

        # Handle exact value - swept parameters can't match exact values
        # Strings are iterable, but are single values
        if isinstance(value, str) or not isinstance(value, Iterable):
            return False

        # Handle range
        min_val, max_val = self._validate_parameter_range(name, value)

        # For swept parameter, check if sweep range matches exactly or belongs to it
        if exact_sweep:
            return param.state.min_val == min_val and param.state.max_val == max_val
        else:
            return param.state.min_val >= min_val and param.state.max_val <= max_val

    def get_parameter(self, name: str) -> Parameter:
        param = self.parameters.get(name)
        if not param:
            raise ValueError(
                f'{name} parameter is not defined for file {self.path.name}'
            )
        return param

    def set_parameter_fixed(self, name: str, value: float | str) -> None:
        """
        Set a parameter to a fixed value.

        Args:
            name: Parameter name
            value: Fixed value (numeric or special string value like 'IP', 'OOP')
        """
        param = self.get_parameter(name)
        param.set_fixed(value)

    def set_parameter_swept(
        self,
        name: str,
        min_val: float | str | None = None,
        max_val: float | str | None = None,
    ) -> None:
        """
        Set a parameter as swept.

        Args:
            name: Parameter name
            min_val: Minimum value (None for undefined sweep)
            max_val: Maximum value (None for undefined sweep)

        Note:
            Sweep direction is automatically inferred from min_val and max_val.
            If both are None, creates an undefined sweep (e.g., 'sweepTemp').
        """
        param = self.get_parameter(name)
        param.set_swept(min_val, max_val)

    def state_of(self, name: str) -> ParameterState:
        param = self.get_parameter(name)
        return ParameterState.from_state(param.state)

    def value_of(self, name: str) -> float | None:
        """
        Get the value of a fixed parameter.

        Args:
            name: Parameter name

        Returns:
            Parameter value for fixed parameters, None for swept parameters

        Raises:
            ValueError: If parameter is not defined for this file

        Examples:
            >>> file = MeasurementFile(...)
            >>> temp = file.value_of('temperature')  # Get fixed parameter value
            >>> field = file.value_of('field')  # Returns None if field is swept
        """
        param = self.get_parameter(name)
        if param.state.is_swept:
            return None
        return float(param.state.value) if param.state.value is not None else None

    def range_of(self, name: str) -> tuple[float, float] | None:
        """
        Get the range of a swept parameter.

        Returns (start, end) values preserving the sweep direction from the filename.
        For example, "B14to-14T" returns (14.0, -14.0), not (-14.0, 14.0).

        Args:
            name: Parameter name

        Returns:
            Tuple of (start, end) for swept parameters with defined range,
            None for fixed parameters or swept parameters with undefined range.

        Raises:
            ValueError: If parameter is not defined for this file

        Examples:
            >>> file = MeasurementFile(...)
            >>> # From "B-14to14T" (increasing sweep)
            >>> file.range_of('magnetic_field')  # Returns (-14.0, 14.0)
            >>> # From "B14to-14T" (decreasing sweep)
            >>> file.range_of('magnetic_field')  # Returns (14.0, -14.0)
            >>> # From "T=4.2K" (fixed parameter)
            >>> file.range_of('temperature')  # Returns None
            >>> # From "sweepField" (undefined sweep)
            >>> file.range_of('magnetic_field')  # Returns None
        """
        param = self.get_parameter(name)
        if not param.state.is_swept:
            return None

        min_val = param.state.min_val
        max_val = param.state.max_val

        # Return None if range is undefined
        if min_val is None or max_val is None:
            return None

        # Convert to float and return (start, end) preserving sweep direction
        min_val = float(min_val)
        max_val = float(max_val)

        if param.state.sweep_direction == SweepDirection.DECREASING:
            return (max_val, min_val)
        return (min_val, max_val)

    def add_contacts(
        self,
        contacts: int | tuple | ContactPair | list[int | tuple | ContactPair],
    ) -> None:
        """
        Add new contact pair(s) to the file.

        Args:
            contacts: Contact specification(s) to add:
                - int: single contact (e.g., 1)
                - tuple[2]: contact pair (e.g., (1, 2))
                - tuple[3]: with polarization (e.g., (1, 2, 'I'))
                - tuple[4]: with polarization and magnitude (e.g., (1, 2, 'I', 2e-6))
                - ContactPair: full contact pair object
                - list: any combination of above types

        Raises:
            ValueError: If contact pair already exists

        Examples:
            >>> file.add_contacts(1)  # Add single contact
            >>> file.add_contacts((1, 2))  # Add contact pair
            >>> file.add_contacts((1, 2, 'I', 2e-6))  # Add with polarization and magnitude
            >>> file.add_contacts([(1, 2), (3, 4)])  # Add multiple
        """
        # Normalize to list
        if not isinstance(contacts, list):
            contacts = [contacts]

        # Process each contact specification
        for contact_spec in contacts:
            # Convert to ContactPair if needed
            new_pair = ContactPair.make_from(contact_spec)

            # Check if contact already exists
            for existing_pair in self.contact_pairs:
                if existing_pair.pair_matches(new_pair):
                    raise ValueError(
                        f"Contact pair {new_pair} already exists in file {self.name}"
                    )

            # Add new contact pair
            self.contact_pairs.append(new_pair)

    def update_contacts(
        self,
        contacts: int | tuple | ContactPair | list[int | tuple | ContactPair],
    ) -> None:
        """
        Update existing contact pair(s) - modifies polarization and/or magnitude only.

        Contact numbers (first_contact, second_contact) are NOT updated.
        Matches by contact numbers only, ignoring current polarization/magnitude.
        If multiple pairs exist with the same contacts, only the first is updated.

        Matching is order-sensitive: (1, 2) only matches (1, 2), not (2, 1).

        Args:
            contacts: Contact specification(s) to update:
                - int: single contact (e.g., 1)
                - tuple[2]: contact pair (e.g., (1, 2))
                - tuple[3]: with polarization (e.g., (1, 2, 'I'))
                - tuple[4]: with polarization and magnitude (e.g., (1, 2, 'I', 2e-6))
                - ContactPair: full contact pair object
                - list: any combination of above types

        Raises:
            ValueError: If contact pair not found

        Examples:
            >>> # File has: I1-2(2uA)
            >>> file.update_contacts((1, 2, 'V', 5))  # Updates I1-2(2uA) → V1-2(5V)
            >>> file.update_contacts((1, 2, 'I', None))  # Updates V1-2(5V) → I1-2
            >>> file.update_contacts((1, 2))  # Updates I1-2 → 1-2 (removes polarization)
            >>> # File has: I1-2(2uA) and V1-2(1V)
            >>> file.update_contacts((1, 2, 'I', 3e-6))  # Updates first match only (I1-2)
        """
        # Normalize to list
        if not isinstance(contacts, list):
            contacts = [contacts]

        # Process each contact specification
        for contact_spec in contacts:
            # Convert spec to ContactPair to get new values
            update_pair = ContactPair.make_from(contact_spec)

            # Find matching contact pair using contacts_match (ignores polarization/magnitude)
            found = False
            for i, existing_pair in enumerate(self.contact_pairs):
                if existing_pair.contacts_match(contact_spec):
                    # Create new ContactPair with updated polarization/magnitude
                    # but keep original contact numbers
                    self.contact_pairs[i] = ContactPair(
                        first_contact=existing_pair.first_contact,
                        second_contact=existing_pair.second_contact,
                        polarization=update_pair.polarization,
                        magnitude=update_pair.magnitude,
                    )
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Contact pair matching {contact_spec} not found in file {self.name}"
                )

    def drop_contacts(
        self,
        contacts: int | tuple | list[int | tuple] | str,
    ) -> None:
        """
        Remove contact pair(s) from the file.

        Uses progressive matching to find contacts:
        - tuple[2]: matches by contacts only (first match if ambiguous)
        - tuple[3]: matches by contacts + polarization
        - tuple[4]: matches exactly (all fields)

        Matching is order-sensitive: (1, 2) only matches (1, 2), not (2, 1).

        Args:
            contacts: Contact specification(s) to remove:
                - int: single contact (e.g., 1)
                - tuple[2]: contact pair (e.g., (1, 2))
                - tuple[3]: with polarization (e.g., (1, 2, 'I'))
                - tuple[4]: with polarization and magnitude (e.g., (1, 2, 'I', 2e-6))
                - list: list of ints or tuples
                - 'all': special string to remove all contacts

        Raises:
            ValueError: If contact not found

        Examples:
            >>> # File has: I1-2(2uA) and V1-2(1V)
            >>> file.drop_contacts((1, 2, 'V'))  # Drops V1-2(1V) only
            >>> file.drop_contacts((1, 2))  # Drops first match (I1-2 in this case)
            >>> file.drop_contacts([(3, 4), (5, 6)])  # Drop multiple
            >>> file.drop_contacts('all')  # Drop all contacts
        """
        # Handle 'all' special case
        if contacts == 'all':
            self.contact_pairs.clear()
            return

        # Normalize to list
        if not isinstance(contacts, list):
            contacts = [contacts]

        # Process each contact specification
        for contact_spec in contacts:
            # Find and remove matching contact pair using progressive matching
            found = False
            for i, existing_pair in enumerate(self.contact_pairs):
                if existing_pair.pair_matches(contact_spec):
                    self.contact_pairs.pop(i)
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Contact pair matching {contact_spec} not found in file {self.name}"
                )

    def _parse_filename(self) -> None:
        # Get filename without extension
        name = self.path.stem
        # Split by separator
        name_parts = name.split(self.separator)

        for part in name_parts:
            self._parse_part(part)

    def _parse_part(self, part: str) -> None:
        # Try to parse as contact pair first
        contact_pair = ContactPair.from_string(part)
        if contact_pair is not None:
            self.contact_pairs.append(contact_pair)
            return

        # Try to parse as parameter
        for param_def in self.param_definitions.values():
            # Try as sweep
            param_name = param_def.name_id
            param = Parameter(param_def)
            is_sweep = param.parse_sweep(part)
            if is_sweep:
                try:
                    self.parameters[param_name].update(param)
                except KeyError:
                    self.parameters[param_name] = param
                continue
            is_fixed = param.parse_fixed(part)
            if is_fixed:
                try:
                    self.parameters[param_name].update(param)
                except KeyError:
                    self.parameters[param_name] = param
                continue

    def merge(
        self,
        other: 'MeasurementFile',
        strict_mode: bool = False,
    ) -> 'MeasurementFile':
        """
        Merge this file representation with another one.

        Args:
            other: Another FileRepresentation to merge with
            strict_mode: If True, verify all parameters are equal before merging

        Returns:
            A new MeasurementFile with merged parameters and contact pairs

        Raises:
            ValueError: If strict_mode is True and parameters differ between files
        """
        # Check parameters in strict mode
        if strict_mode:
            for param_name, param in self.parameters.items():
                if param_name in other.parameters:
                    other_param = other.parameters[param_name]
                    # Check if parameters are equal
                    if param.state != other_param.state:
                        raise ValueError(
                            f"Parameter '{param_name}' differs between files in strict mode"
                        )

        # Merge parameters (self take precedence in case of conflict)
        merged_parameters = {}
        merged_parameters.update(other.parameters)
        merged_parameters.update(self.parameters)
        parameters_list = [param for param in merged_parameters.values()]

        # Merge contact pairs (removing duplicates)
        merged_contacts = []
        seen_contacts = set()

        # Add contacts from self
        for contact in self.contact_pairs:
            key = contact.to_tuple()
            if key not in seen_contacts:
                merged_contacts.append(contact.copy())
                seen_contacts.add(key)

        # Add contacts from other
        for contact in other.contact_pairs:
            key = contact.to_tuple()
            if key not in seen_contacts:
                merged_contacts.append(contact.copy())
                seen_contacts.add(key)

        # Create a new MeasurementFile with merged data
        # Use the separator from the current instance
        merged_file = type(self)(
            path=self.path, parameters=parameters_list, separator=self.separator
        )
        merged_file.contact_pairs = merged_contacts

        merged_filename = merged_file.generate_filename()
        merged_file.path = merged_file.path.parent / merged_filename
        return merged_file

    def rename(
        self,
        directory: str | Path | None = None,
        name: str | Path | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        sep: str | None = None,
        ext: str | None = None,
    ) -> None:
        # Change separator
        if sep:
            self.separator = sep

        # Generate new path
        self.path = self._generate_path(
            directory=directory,
            name=name,
            prefix=prefix,
            postfix=postfix,
            sep=sep,
            ext=ext,
        )

    def _generate_path(
        self,
        directory: str | Path | None = None,
        name: str | Path | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        sep: str | None = None,
        ext: str | None = None,
    ) -> Path:
        if not directory:
            directory = self.path.parent
        directory = Path(directory).expanduser()

        if not name:
            name = self.path.stem

        if sep:
            name = name.replace(self.separator, sep)
            self.separator = sep

        if prefix:
            name = self.separator.join([prefix, name])

        if postfix:
            name = self.separator.join([name, postfix])

        if ext and not ext.startswith('.'):
            ext = f".{ext}"
        elif ext is None:
            ext = self.path.suffix
        else:
            ext = ''

        return directory / f"{name}{ext}"

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new = type(self)(
            path=self.path,
            parameters=[param for param in self.parameters.values()],
            separator=self.separator,
        )
        new.contact_pairs = [pair.copy() for pair in self.contact_pairs]
        return new

    def generate_filename(
        self, prefix: str = None, postfix: str = None, sep: str = None, ext: str = None
    ) -> str:
        """
        Generate a filename based on stored parameters and contact pairs.

        Args:
            prefix: Optional prefix for the filename
            postfix: Optional postfix for the filename
            sep: Optional separator (instance separator if None)
            ext: File extension (instance extension if None)

        Returns:
            A string representing the new filename
        """
        # Use instance separator if not provided
        sep = sep if sep is not None else self.separator

        # Build the contact part of the filename
        contact_parts = []
        for contacts in self.contact_pairs:
            contact_parts.append(str(contacts))

        # Determine parameters order
        parameters_ordered = []
        # Add sweeping parameters
        for param in self.parameters.values():
            if param.state.is_swept:
                parameters_ordered.append(param)
        # Add fixed parameters
        for param in self.parameters.values():
            if not param.state.is_swept:
                parameters_ordered.append(param)

        param_parts = []
        # Build the parameter part of the filename
        for param in parameters_ordered:
            param_parts.append(str(param))

        # Combine all parts
        filename_parts = []
        if prefix:
            filename_parts.append(prefix)
        filename_parts.extend(contact_parts)
        filename_parts.extend(param_parts)
        if postfix:
            filename_parts.append(postfix)

        # Join with separator and add extension
        filename = sep.join(filename_parts)

        # Add extension if it doesn't already have one
        if ext and not ext.startswith('.'):
            ext = f".{ext}"
        elif ext is None:
            ext = self.path.suffix
        else:
            ext = ''

        return f"{filename}{ext}"
