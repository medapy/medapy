class UnitOverwriteWarning(UserWarning):
    pass


class MagnitudeFormatWarning(UserWarning):
    """Warning issued when a contact pair string contains parentheses
    indicating a magnitude value, but the magnitude fails to parse.

    This typically occurs when:
    - Multiple SI prefixes are used (e.g., 'nmA' instead of 'nA')
    - Invalid SI prefix is used (e.g., 'XA')
    - Unit is missing (e.g., '100' instead of '100A')

    The contact pair and polarization are still parsed successfully,
    but the magnitude will be None.
    """
    pass