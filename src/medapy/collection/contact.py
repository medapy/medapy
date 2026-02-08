from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import re
import warnings

from medapy.utils.prefixes import SYMBOL_TO_MULTIPLIER, format_with_prefix
from medapy.utils.warnings import MagnitudeFormatWarning

class PolarizationType(Enum):
    CURRENT = 'I'
    VOLTAGE = 'V'

    __aliases_current = frozenset(('i', 'current'))
    __aliases_voltage = frozenset(('v', 'voltage'))

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in cls.__aliases_current:
                return cls.CURRENT
            elif value in cls.__aliases_voltage:
                return cls.VOLTAGE

    def __eq__(self, other):
        if isinstance(other, str):
            other = self._missing_(other)
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


# Dynamically generate SI prefix character class from supported prefixes
_SI_PREFIX_CHARS = ''.join(re.escape(p) for p in SYMBOL_TO_MULTIPLIER.keys())

contact_pattern = re.compile(
    rf'([IV])(\d+)(?:-(\d+))?(?:\((-?\d+\.?\d*(?:[eE][+-]?\d+)?[{_SI_PREFIX_CHARS}]?[AV])\))?'
)


@dataclass(frozen=True)
class ContactPair:
    """
    Immutable representation of an electrical contact pair configuration.

    Stores contact numbers, polarization type (current/voltage), and magnitude.
    Used for identifying and filtering measurement files by contact configuration.

    Parameters
    ----------
    first_contact : int
        First contact number (required).
    second_contact : int, optional
        Second contact number. None for single-contact measurements.
    polarization : PolarizationType or str, optional
        Polarization type: 'I' for current, 'V' for voltage.
        Accepts string or PolarizationType enum.
    magnitude : Decimal, float, int, or str, optional
        Polarization magnitude. Automatically converted to Decimal.

    Attributes
    ----------
    first_contact : int
        First contact number.
    second_contact : int or None
        Second contact number, or None for single contacts.
    polarization : PolarizationType or None
        Polarization type (CURRENT or VOLTAGE).
    magnitude : Decimal or None
        Polarization magnitude.

    Examples
    --------
    >>> # Create contact pair with current polarization
    >>> pair = ContactPair(1, 5, 'I', 10e-6)  # I1-5(10uA)
    >>> # Single contact
    >>> single = ContactPair(3)
    >>> # Voltage polarization
    >>> vpair = ContactPair(2, 6, 'V', 1e-3)  # V2-6(1mV)
    >>> # String representation
    >>> str(pair)  # 'I1-5(1.0e-05A)'
    >>> # Parsing from string
    >>> parsed = ContactPair.from_string("I1-5(10uA)")
    >>> # Progressive matching
    >>> pair.pair_matches((1, 5))  # True (matches contacts)
    >>> pair.pair_matches((1, 5, 'I'))  # True (matches contacts + polarization)

    See Also
    --------
    make_from : Create ContactPair from various input formats
    from_string : Parse from string representation
    pair_matches : Progressive matching based on input type
    contacts_match : Match only contact numbers
    """
    # For single contact, second_contact will be None
    first_contact: int | None = None
    second_contact: int | None = None
    polarization: PolarizationType | None = None
    magnitude: Decimal | None = None

    def __copy__(self):
        return type(self)(
            first_contact=self.first_contact,
            second_contact=self.second_contact,
            polarization=self.polarization,
            magnitude=self.magnitude,
        )

    @classmethod
    def make_from(cls, data: int | tuple | str | "ContactPair") -> "ContactPair":
        """
        Create ContactPair from various input formats.

        Parameters
        ----------
        data : int, tuple, str, or ContactPair
            Contact specification:
            - int: single contact (e.g., 1)
            - tuple[2]: (first_contact, second_contact)
            - tuple[3]: (first_contact, second_contact, polarization)
            - tuple[4]: (first_contact, second_contact, polarization, magnitude)
            - str: string representation (e.g., "I1-2(2uA)", "V3-4", "1")
            - ContactPair: returns as-is

        Returns
        -------
        ContactPair
            ContactPair instance created from input.

        Raises
        ------
        ValueError
            If invalid input type or string parsing fails, or if tuple
            length is greater than 4.

        Examples
        --------
        >>> # From integer (single contact)
        >>> ContactPair.make_from(1)
        >>> # From tuple
        >>> ContactPair.make_from((1, 5))
        >>> ContactPair.make_from((1, 5, 'I'))
        >>> ContactPair.make_from((1, 5, 'I', 10e-6))
        >>> # From string
        >>> ContactPair.make_from("I1-5(10uA)")
        >>> ContactPair.make_from("V2-6")
        >>> # From existing ContactPair (pass-through)
        >>> pair = ContactPair(1, 5)
        >>> ContactPair.make_from(pair)  # Returns same instance

        See Also
        --------
        from_string : Parse specifically from string
        """
        # ContactPair: pass through
        if isinstance(data, ContactPair):
            return data

        # String: parse using from_string
        if isinstance(data, str):
            parsed = cls.from_string(data)
            if parsed is None:
                raise ValueError(f"Failed to parse contact pair from string: '{data}'")
            return parsed

        # Int: single contact
        if isinstance(data, int):
            return cls(first_contact=data)

        # Tuple: unpack into constructor
        if isinstance(data, tuple):
            if len(data) <= 4:
                return cls(*data)
            else:
                raise ValueError(f"Tuple length must be <= 4, got {len(data)}")

        # Unknown type
        raise ValueError(f"Expected int, tuple, str, or ContactPair, got {type(data)}")

    def __str__(self) -> str:
        result = f"{self.first_contact}"
        if self.second_contact is not None:
            result += f"-{self.second_contact}"

        if self.polarization is not None:
            result = f"{self.polarization.value}" + result

            if self.magnitude is not None:
                if self.polarization == PolarizationType.CURRENT:
                    unit = 'A'
                else:
                    unit = 'V'
                # Format magnitude with SI prefix (no space for compact representation)
                formatted_magnitude = format_with_prefix(
                    float(self.magnitude),
                    precision='auto',
                    use_space=False
                )
                result += f"({formatted_magnitude}{unit})"
        return result

    def __post_init__(self):
        # Validate first_contact is not None
        if self.first_contact is None:
            raise ValueError("first_contact cannot be None")

        # Convert types (use object.__setattr__ for frozen dataclass)
        if isinstance(self.polarization, str):
            object.__setattr__(self, 'polarization', PolarizationType(self.polarization))
        if isinstance(self.magnitude, (str, int, float)):
            object.__setattr__(self, 'magnitude', Decimal(str(self.magnitude)))

    @classmethod
    def from_string(cls, text: str) -> "ContactPair | None":
        """
        Parse contact pair from string.

        Args:
            text: String representation (e.g., "I1-2(2uA)", "V3-4", "V1")

        Returns:
            ContactPair instance if parsing succeeds, None otherwise

        Warnings:
            MagnitudeFormatWarning: If text contains parentheses but magnitude
                fails to parse (e.g., multiple prefixes, invalid prefix, missing unit)
        """
        m = contact_pattern.match(text)
        if not m:
            return None
        type_str, first, second, magnitude = m.groups()

        # Create the contact pair
        contact_pair = cls(
            first_contact=int(first),
            second_contact=int(second) if second else None,
            polarization=PolarizationType(type_str),
            magnitude=cls._convert_magntude(magnitude) if magnitude else None,
        )

        # Check if magnitude was attempted but failed to parse
        has_parens = '(' in text and ')' in text
        if has_parens and contact_pair.magnitude is None:
            warnings.warn(
                f"Invalid magnitude format in '{text}': contact pair parsed as {contact_pair} "
                f"but magnitude could not be parsed. Common issues: invalid prefix "
                f"or missing unit (A/V).",
                MagnitudeFormatWarning,
                stacklevel=2
            )

        return contact_pair

    def pair_matches(self, pair: int | tuple | str | ContactPair) -> bool:
        """
        Progressive matching based on input type.

        Args:
            pair: Match specification
                - int: match single contact (first_contact=n, second_contact=None)
                - tuple[2]: match contacts only
                - tuple[3]: match contacts + polarization
                - tuple[4]: match contacts + polarization + magnitude
                - str: string representation (e.g., "I1-2(2uA)", "V3-4", "V1")
                - ContactPair: exact equality check (all fields must match)

        Returns:
            bool: True if matches according to progressive rules
        """
        # Handle string: convert to ContactPair
        if isinstance(pair, str):
            pair = ContactPair.make_from(pair)
            return self == pair

        # Handle ContactPair: exact equality
        if isinstance(pair, ContactPair):
            return self == pair

        # Handle int: single contact
        if isinstance(pair, int):
            return self.first_contact == pair and self.second_contact is None

        # Handle tuple: progressive matching
        if isinstance(pair, tuple):
            pair_len = len(pair)

            if pair_len == 2:
                # Match contacts only
                first, second = pair
                return self.first_contact == first and self.second_contact == second

            elif pair_len == 3:
                # Match contacts + polarization
                first, second, polarization = pair
                return (
                    self.first_contact == first
                    and self.second_contact == second
                    and self.polarization == polarization
                )

            elif pair_len == 4:
                # Match contacts + polarization + magnitude (exact equality)
                return self == type(self)(*pair)

            else:
                raise ValueError(f"Tuple length must be 2-4, got {pair_len}")

        raise TypeError(f"Expected int, tuple, str, or ContactPair, got {type(pair)}")

    def contacts_match(self, other: int | tuple | str | ContactPair) -> bool:
        """
        Match only contact numbers, ignoring polarization and magnitude.

        Args:
            other: Contact specification
                - int: single contact
                - tuple: contact pair (uses first 1-2 elements)
                - str: string representation (e.g., "I1-2(2uA)", "V3-4", "V1")
                - ContactPair: compare contact numbers directly

        Returns:
            bool: True if contact numbers match

        Examples:
            >>> pair = ContactPair(1, 2, 'I', 1e-6)
            >>> pair.contacts_match(ContactPair(1, 2, 'V', 2e-6))  # True
            >>> pair.contacts_match((1, 2))  # True
            >>> pair.contacts_match((1, 2, 'V'))  # True (ignores polarization)
            >>> pair.contacts_match("I1-2")  # True (ignores polarization)
            >>> pair.contacts_match((2, 1))  # False (order matters)
        """
        # Handle string: convert to ContactPair
        if isinstance(other, str):
            other = ContactPair.make_from(other)
            return (
                self.first_contact == other.first_contact
                and self.second_contact == other.second_contact
            )

        if isinstance(other, int):
            return self.first_contact == other and self.second_contact is None

        if isinstance(other, ContactPair):
            return (
                self.first_contact == other.first_contact
                and self.second_contact == other.second_contact
            )

        if isinstance(other, tuple):
            if len(other) == 0:
                raise ValueError("Empty tuple not allowed")
            first = other[0]
            second = other[1] if len(other) >= 2 else None
            return self.first_contact == first and self.second_contact == second

        raise TypeError(f"Expected int, tuple, str, or ContactPair, got {type(other)}")

    def to_tuple(self):
        return (
            self.first_contact,
            self.second_contact,
            self.polarization,
            self.magnitude,
        )

    def copy(self):
        return self.__copy__()

    def polarized(self, polarization, magnitude=None):
        return type(self)(
            self.first_contact,
            self.second_contact,
            polarization=polarization,
            magnitude=magnitude,
        )

    @staticmethod
    def _convert_magntude(magnitude):
        # Strip unit suffix (A or V) before parsing
        magnitude = magnitude.rstrip('AV')

        # Check if there's a prefix (last character is a letter)
        if magnitude and magnitude[-1].isalpha():
            prefix = magnitude[-1]
            numeric_part = magnitude[:-1]

            if prefix in SYMBOL_TO_MULTIPLIER:
                # Use Decimal throughout to avoid float precision issues
                multiplier = SYMBOL_TO_MULTIPLIER[prefix]
                value = Decimal(numeric_part) * Decimal(str(multiplier))
                return value
            else:
                raise ValueError(
                    f"Unknown prefix: '{prefix}'. "
                    f"Valid prefixes: {list(SYMBOL_TO_MULTIPLIER.keys())}"
                )
        else:
            # No prefix, parse directly as Decimal
            return Decimal(magnitude)

    def __hash__(self):
        return hash(
            (self.first_contact, self.second_contact, self.polarization, self.magnitude)
        )
