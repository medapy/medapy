"""Tests for ContactPair class."""

import pytest
from decimal import Decimal
from medapy.collection.mfile import ContactPair, PolarizationType


class TestContactPairCreation:
    """Test ContactPair creation from different inputs."""

    def test_create_from_int(self):
        """Test creating ContactPair from single integer."""
        pair = ContactPair.make_from(5)
        assert pair.first_contact == 5
        assert pair.second_contact is None
        assert pair.polarization is None
        assert pair.magnitude is None

    def test_create_from_tuple_two_elements(self):
        """Test creating ContactPair from 2-element tuple."""
        pair = ContactPair.make_from((1, 5))
        assert pair.first_contact == 1
        assert pair.second_contact == 5
        assert pair.polarization is None
        assert pair.magnitude is None

    def test_create_from_tuple_three_elements(self):
        """Test creating ContactPair from 3-element tuple."""
        pair = ContactPair.make_from((1, 5, 'I'))
        assert pair.first_contact == 1
        assert pair.second_contact == 5
        assert pair.polarization == PolarizationType.CURRENT
        assert pair.magnitude is None

    def test_create_from_tuple_four_elements(self):
        """Test creating ContactPair from 4-element tuple."""
        pair = ContactPair.make_from((1, 5, 'V', 0.0001))
        assert pair.first_contact == 1
        assert pair.second_contact == 5
        assert pair.polarization == PolarizationType.VOLTAGE
        assert pair.magnitude == Decimal('0.0001')

    def test_create_from_contact_pair(self):
        """Test creating ContactPair from another ContactPair."""
        original = ContactPair(1, 5, 'I', Decimal('0.00001'))
        pair = ContactPair.make_from(original)
        assert pair is original  # Should return the same instance

    def test_create_invalid_tuple_length(self):
        """Test that tuple with >4 elements raises ValueError."""
        with pytest.raises(ValueError, match="Tuple length must be <= 4"):
            ContactPair.make_from((1, 2, 3, 4, 5))

    def test_create_invalid_type(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Expected int, tuple, str, or ContactPair"):
            ContactPair.make_from([1, 2])  # List instead of tuple

    def test_first_contact_cannot_be_none(self):
        """Test that first_contact=None raises ValueError."""
        with pytest.raises(ValueError, match="first_contact cannot be None"):
            ContactPair(first_contact=None)


class TestContactPairStringParsing:
    """Test parsing ContactPair from strings."""

    @pytest.mark.parametrize("input_str,expected", [
        # Simple contact pairs
        ("I1-5", ContactPair(1, 5, 'I', None)),
        ("V2-3", ContactPair(2, 3, 'V', None)),

        # Single contacts
        ("I1", ContactPair(1, None, 'I', None)),
        ("V20", ContactPair(20, None, 'V', None)),

        # With magnitudes in various units
        ("I1-5(10mA)", ContactPair(1, 5, 'I', Decimal('0.01'))),
        ("V2-3(100uV)", ContactPair(2, 3, 'V', Decimal('0.0001'))),
        ("I1-2(1uA)", ContactPair(1, 2, 'I', Decimal('0.000001'))),
        ("V3-4(2nV)", ContactPair(3, 4, 'V', Decimal('2e-9'))),
        ("I5-6(500pA)", ContactPair(5, 6, 'I', Decimal('500e-12'))),

        # With larger units
        ("V1-2(1kV)", ContactPair(1, 2, 'V', Decimal('1000'))),
        ("I3-4(2MA)", ContactPair(3, 4, 'I', Decimal('2000000'))),
    ])
    def test_from_string_valid(self, input_str, expected):
        """Test parsing valid contact pair strings."""
        result = ContactPair.from_string(input_str)
        assert result == expected

    def test_from_string_invalid(self):
        """Test that invalid strings return None."""
        assert ContactPair.from_string("invalid") is None
        assert ContactPair.from_string("") is None
        assert ContactPair.from_string("1-2") is None  # Missing polarization

    def test_make_from_string(self):
        """Test make_from with string input."""
        pair = ContactPair.make_from("I1-5(10mA)")
        assert pair.first_contact == 1
        assert pair.second_contact == 5
        assert pair.polarization == PolarizationType.CURRENT
        assert pair.magnitude == Decimal('0.01')

    def test_make_from_invalid_string(self):
        """Test that make_from raises ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Failed to parse contact pair"):
            ContactPair.make_from("invalid_string")


class TestContactPairMagnitudeConversion:
    """Test SI prefix conversion in magnitudes."""

    @pytest.mark.parametrize("magnitude_str,expected_decimal", [
        ("10mA", Decimal('0.01')),
        ("100uV", Decimal('0.0001')),
        ("1uA", Decimal('0.000001')),
        ("2nV", Decimal('2e-9')),
        ("500pA", Decimal('500e-12')),
        ("3fA", Decimal('3e-15')),
        ("1kV", Decimal('1000')),
        ("2MA", Decimal('2000000')),
        ("5GV", Decimal('5e9')),
        ("1TV", Decimal('1e12')),
    ])
    def test_magnitude_conversion(self, magnitude_str, expected_decimal):
        """Test conversion of magnitude strings with SI prefixes."""
        result = ContactPair._convert_magntude(magnitude_str)
        assert result == expected_decimal


class TestContactPairMatching:
    """Test pair_matches and contacts_match methods."""

    def test_pair_matches_with_int(self):
        """Test pair_matches with integer (single contact)."""
        pair = ContactPair(5, None, 'I', None)
        assert pair.pair_matches(5) is True

        pair_double = ContactPair(1, 5, 'I', None)
        assert pair_double.pair_matches(1) is False  # Has second contact

    def test_pair_matches_with_tuple_two(self):
        """Test pair_matches with 2-element tuple (contacts only)."""
        pair = ContactPair(1, 5, 'I', Decimal('0.01'))
        assert pair.pair_matches((1, 5)) is True
        assert pair.pair_matches((5, 1)) is False
        assert pair.pair_matches((1, 2)) is False

    def test_pair_matches_with_tuple_three(self):
        """Test pair_matches with 3-element tuple (contacts + polarization)."""
        pair = ContactPair(1, 5, 'I', Decimal('0.01'))
        assert pair.pair_matches((1, 5, 'I')) is True
        assert pair.pair_matches((1, 5, 'V')) is False
        assert pair.pair_matches((1, 5, PolarizationType.CURRENT)) is True

    def test_pair_matches_with_tuple_four(self):
        """Test pair_matches with 4-element tuple (exact match)."""
        pair = ContactPair(1, 5, 'I', Decimal('0.01'))
        assert pair.pair_matches((1, 5, 'I', Decimal('0.01'))) is True
        assert pair.pair_matches((1, 5, 'I', Decimal('0.02'))) is False
        assert pair.pair_matches((1, 5, 'I', 0.01)) is True  # Float converted to Decimal

    def test_pair_matches_with_string(self):
        """Test pair_matches with string."""
        pair = ContactPair(1, 5, 'I', Decimal('0.01'))
        assert pair.pair_matches("I1-5(10mA)") is True
        assert pair.pair_matches("I1-5") is False  # Missing magnitude
        assert pair.pair_matches("V1-5(10mA)") is False

    def test_pair_matches_with_contact_pair(self):
        """Test pair_matches with another ContactPair."""
        pair1 = ContactPair(1, 5, 'I', Decimal('0.01'))
        pair2 = ContactPair(1, 5, 'I', Decimal('0.01'))
        pair3 = ContactPair(1, 5, 'V', Decimal('0.01'))
        assert pair1.pair_matches(pair2) is True
        assert pair1.pair_matches(pair3) is False

    def test_contacts_match_ignores_polarization(self):
        """Test that contacts_match ignores polarization and magnitude."""
        pair = ContactPair(1, 2, 'I', Decimal('0.000001'))

        # Same contacts, different polarization
        assert pair.contacts_match(ContactPair(1, 2, 'V', Decimal('0.0002'))) is True

        # Tuple with polarization - should still match
        assert pair.contacts_match((1, 2, 'V')) is True
        assert pair.contacts_match((1, 2)) is True

        # String with different polarization
        assert pair.contacts_match("V1-2") is True
        assert pair.contacts_match("I1-2") is True

        # Different contacts - should not match
        assert pair.contacts_match((2, 1)) is False

    def test_contacts_match_with_single_contact(self):
        """Test contacts_match with single contact."""
        pair = ContactPair(5, None, 'I', None)
        assert pair.contacts_match(5) is True
        assert pair.contacts_match((5,)) is True
        assert pair.contacts_match("I5") is True
        assert pair.contacts_match(6) is False

    def test_contacts_match_empty_tuple_raises(self):
        """Test that empty tuple raises ValueError."""
        pair = ContactPair(1, 2, 'I', None)
        with pytest.raises(ValueError, match="Empty tuple not allowed"):
            pair.contacts_match(())


class TestContactPairStringRepresentation:
    """Test string representation of ContactPair."""

    def test_str_single_contact_no_polarization(self):
        """Test __str__ for single contact without polarization."""
        pair = ContactPair(5)
        assert str(pair) == "5"

    def test_str_contact_pair_no_polarization(self):
        """Test __str__ for contact pair without polarization."""
        pair = ContactPair(1, 5)
        assert str(pair) == "1-5"

    def test_str_with_polarization_no_magnitude(self):
        """Test __str__ with polarization but no magnitude."""
        pair = ContactPair(1, 5, 'I')
        assert str(pair) == "I1-5"

    def test_str_with_polarization_and_magnitude(self):
        """Test __str__ with polarization and magnitude."""
        # Small magnitude
        pair = ContactPair(1, 5, 'I', Decimal('0.00001'))
        result = str(pair)
        assert result.startswith("I1-5(")
        assert "A)" in result

        # Voltage
        pair_v = ContactPair(2, 3, 'V', Decimal('0.0001'))
        result_v = str(pair_v)
        assert result_v.startswith("V2-3(")
        assert "V)" in result_v


class TestContactPairOtherMethods:
    """Test other ContactPair methods."""

    def test_to_tuple(self):
        """Test to_tuple conversion."""
        pair = ContactPair(1, 5, 'I', Decimal('0.01'))
        result = pair.to_tuple()
        assert result == (1, 5, PolarizationType.CURRENT, Decimal('0.01'))

    def test_copy(self):
        """Test copy method."""
        original = ContactPair(1, 5, 'I', Decimal('0.01'))
        copied = original.copy()
        assert copied == original
        assert copied is not original  # Different instances

    def test_polarized(self):
        """Test polarized method creates new instance with polarization."""
        original = ContactPair(1, 5)
        polarized = original.polarized('I', Decimal('0.01'))

        assert polarized.first_contact == 1
        assert polarized.second_contact == 5
        assert polarized.polarization == PolarizationType.CURRENT
        assert polarized.magnitude == Decimal('0.01')

        # Original unchanged
        assert original.polarization is None
        assert original.magnitude is None
