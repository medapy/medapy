"""
Tests for optional dependency handling.

This module tests the guarded import helper in medapy.utils.optional and its
use by functions that depend on packages outside the core install.
"""

import numpy as np
import pytest

from medapy.utils import misc
from medapy.utils.optional import require


class TestRequire:
    def test_returns_installed_module(self):
        mod = require('math', extra='fit')
        assert mod.sqrt(4) == 2

    def test_returns_installed_submodule(self):
        mod = require('os.path', extra='fit')
        assert mod.basename('/a/b.txt') == 'b.txt'

    def test_raises_import_error_when_missing(self):
        with pytest.raises(ImportError, match=r"nonexistent_module_xyz"):
            require('nonexistent_module_xyz', extra='fit')

    def test_error_message_names_the_extra(self):
        with pytest.raises(ImportError, match=r"pip install medapy\[fit\]"):
            require('nonexistent_module_xyz', extra='fit')

    def test_error_chains_original_import_error(self):
        with pytest.raises(ImportError) as exc_info:
            require('nonexistent_module_xyz', extra='fit')
        assert isinstance(exc_info.value.__cause__, ImportError)


class TestSavgolRequiresScipy:
    def test_works_when_scipy_available(self):
        result = misc.savgol_filter(np.arange(9.0), 3, order=1)
        assert result.shape == (9,)
        assert not np.isnan(result).all()

    def test_raises_import_error_when_scipy_missing(self, monkeypatch):
        def missing(module, extra):
            raise ImportError(f"'{module}' is required but is not installed. "
                              f"Install it with: pip install medapy[{extra}]")

        monkeypatch.setattr(misc, 'require', missing)
        with pytest.raises(ImportError, match=r"pip install medapy\[fit\]"):
            misc.savgol_filter(np.arange(9.0), 3, order=1)
