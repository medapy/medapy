"""
Tests for deprecated import paths kept as back-compat shims.

This module tests that the old medapy.utils.misc path still resolves to the
functions that moved to medapy.analysis.processing and medapy.utils.validations.
"""

import numpy as np
import pytest

from medapy.analysis import processing
from medapy.utils import misc, validations


class TestMiscShim:
    def test_public_function_forwards_to_processing(self):
        with pytest.deprecated_call():
            func = misc.symmetrize
        assert func is processing.symmetrize

    def test_forwarded_function_still_works(self):
        with pytest.deprecated_call():
            result = misc.normalize(np.array([1.0, 2.0, 4.0]), by='first')
        np.testing.assert_allclose(result, [1.0, 2.0, 4.0])

    @pytest.mark.parametrize('old_name, new_func', [
        ('_validate_option', validations.validate_option),
        ('_validate_xy', validations.validate_xy),
    ])
    def test_moved_validators_forward_to_validations(self, old_name, new_func):
        with pytest.deprecated_call():
            func = getattr(misc, old_name)
        assert func is new_func

    def test_warning_names_the_new_location(self):
        with pytest.warns(DeprecationWarning, match=r'medapy\.analysis\.processing\.interpolate'):
            misc.interpolate

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match='no_such_function'):
            misc.no_such_function

    def test_dir_lists_forwarded_names(self):
        names = dir(misc)
        assert 'symmetrize' in names
        assert '_validate_option' in names
