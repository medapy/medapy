"""Shared pytest fixtures for MeDaPy tests."""

import pytest
import pandas as pd
from medapy import ureg


@pytest.fixture
def unit_registry():
    """Pint unit registry."""
    return ureg


@pytest.fixture
def simple_df():
    """Basic DataFrame with units in column names."""
    return pd.DataFrame({
        'Field (T)': [0, 1, 2, 3],
        'Resistance (Ohm)': [100, 110, 120, 130]
    })


@pytest.fixture
def sample_filenames():
    """Realistic measurement filenames for testing."""
    return [
        "sample_I1-5(10mA)_V20-21_sweepField_T=4.2K_Rxx.csv",
        "sample_V2-3_B-14to14T_T=1.8K_Rxy.csv",
        "device_I1-2(1uA)_V3-4_B=0T_T=300K_Rxx.csv",
    ]
