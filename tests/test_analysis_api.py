"""
Tests for the public import surface of medapy.analysis.

This module tests that domain packages are reachable as short aliases on
medapy.analysis, that touching an alias registers the domain's pandas accessor,
and that importing medapy.analysis does not pull in a domain's optional
dependencies.
"""

import subprocess
import sys

import pytest

import medapy.analysis
from medapy.analysis.electron_transport import transport


class TestEtrAlias:
    def test_alias_resolves_to_functional_module(self):
        assert medapy.analysis.etr is transport

    def test_alias_exposes_functional_api(self):
        assert callable(medapy.analysis.etr.fit_twoband)

    def test_from_import_works(self):
        from medapy.analysis import etr
        assert etr is transport

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match='no_such_domain'):
            medapy.analysis.no_such_domain

    def test_dir_lists_lazy_domains(self):
        assert 'etr' in dir(medapy.analysis)


class TestLazyDomainImport:
    """Each case needs a fresh interpreter: this one has already imported the domain."""

    @staticmethod
    def _run(body: str) -> int:
        return subprocess.run([sys.executable, '-c', f'import sys\n{body}']).returncode

    def test_importing_analysis_does_not_import_lmfit(self):
        code = ('import medapy.analysis\n'
                'sys.exit(1 if "lmfit" in sys.modules else 0)')
        assert self._run(code) == 0

    def test_importing_analysis_does_not_register_etr(self):
        code = ('import pandas as pd\n'
                'import medapy.analysis\n'
                'sys.exit(1 if hasattr(pd.DataFrame, "etr") else 0)')
        assert self._run(code) == 0

    def test_touching_alias_imports_lmfit_and_registers_etr(self):
        code = ('import pandas as pd\n'
                'import medapy.analysis\n'
                'medapy.analysis.etr\n'
                'sys.exit(0 if "lmfit" in sys.modules and hasattr(pd.DataFrame, "etr") else 1)')
        assert self._run(code) == 0
