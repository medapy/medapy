# MeDaPy

**Me**asurement **Da**ta **Py**thon package: unit-aware pandas tools for measured data, with a focus on electron transport.

- **Measurement sheets** (`df.ms`): units parsed from column names, labels and axes for convenient access, unit conversion and unit-aware arithmetic via Pint.
- **Collections**: build a set of measurement files from a folder and filter it by parameters parsed from the filenames (temperature, field range, contacts, ...).
- **Processing** (`df.proc`): range selection, symmetrization, normalization, interpolation, smoothing.
- **Electron transport** (`df.etr`): resistance to resistivity, linear Hall and two-band/multiband fits.

!!! note
    The package and its documentation are in alpha; the API may still change.

## Installation

```bash
pip install medapy          # core: numpy, pandas, pint, pint-pandas
pip install "medapy[fit]"   # + scipy, lmfit, numdifftools (needed for df.etr and fitting)
pip install "medapy[plot]"  # + matplotlib (publication style and colormaps)
pip install "medapy[all]"   # fit + plot
```

MeDaPy requires Python 3.11 or newer.

## Quickstart

```python
--8<-- "examples/0_quickstart.py:quickstart"
```

The printed sheet shows each column's unit (`U`), axis (`A`) and labels (`L`) above the data.

## Where to go next

- [Tutorials](tutorials.md): a guided path through five example scripts, from measurement-sheet basics to a complete fit-and-save workflow.
- [Reference](reference.md): the API, generated from the docstrings.
