# Fit data

Polynomial fits work on plain arrays and need only the base install. The transport fits are
on the `df.etr` accessor and need the `fit` extra (`pip install "medapy[fit]"`). All recipes
on this page start from:

```python
--8<-- "examples/how_to/fitting.py:setup"
```

`data` holds the xx and Hall resistance of the same sample and, from them, the resistivities
`rho_xx` and `rho_xy` in ohm·m.

!!! note "Units"
    The transport fits take field in T and resistivity in ohm·m, and return densities in
    m⁻³ and mobilities in m²/(V·s). They do not convert units, so use `convert_unit` first if
    your columns are in other units.

## Fit a polynomial

[`processing.quick_fit`][medapy.analysis.processing.quick_fit] is a least-squares fit of a
polynomial up to `degrees`; it returns the coefficients in ascending powers.
[`processing.make_curve`][medapy.analysis.processing.make_curve] evaluates them. `x_range=`
fits only part of the data.

```python
--8<-- "examples/how_to/fitting.py:polynomial"
```

## Fit chosen powers, including negative ones

Give `degrees` as a list to fit exactly those powers, and pass the same list to
`make_curve`.

```python
--8<-- "examples/how_to/fitting.py:powers"
```

## Linear Hall fit

[`fit_linhall`][medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor.fit_linhall]
is `quick_fit` with degree 1 on a sheet column against field. It returns the coefficients
and a sheet with the fitted line as a new column (`add_col=None` to skip it).

```python
--8<-- "examples/how_to/fitting.py:linhall"
```

## Fit several bands to xx and xy together

[`fit_multiband`][medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor.fit_multiband]
fits one set of carrier densities and mobilities to all given columns at once. `bands` has
one letter per band, `h` for holes and `e` for electrons (`'he'`, `'hee'`, ...), and `p0`
gives the starting values **interleaved**: `[n1, mu1, n2, mu2, ...]`. The data here has a
low-field feature that the band model does not describe, so the fit leaves out
|B| < 6 T. `report=True` prints the lmfit report, with the fitted values in physical units
at the end.

```python
--8<-- "examples/how_to/fitting.py:multiband"
```

On plain arrays:
[`etr.fit_multiband`][medapy.analysis.electron_transport.transport.fit_multiband], which
takes the data as `datasets=[(field, rho, kind), ...]`.

## Constrain or weight a multiband fit

Further keywords go to `etr.fit_multiband`. `bounds` limits parameters (defaults: 1e20 to
1e28 m⁻³ for densities, 0.001 to 1 m²/(V·s) for mobilities), `fix_params` holds them at
their starting values, and `expr` ties one to others. Parameters are fitted as their log10,
so `expr` is written in log10 too: `{'n2': 'n1'}` means equal densities, and
`{'mu2': '0.301 + mu1'}` means `mu2 = 2 * mu1`. `sigmas` weights the columns by their
uncertainties.

```python
--8<-- "examples/how_to/fitting.py:constrain"
```

## Evaluate a model with known parameters

[`calculate_multiband`][medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor.calculate_multiband]
adds model columns for given parameters, in the same interleaved order. For plain arrays,
[`etr.generate_multiband_eq`][medapy.analysis.electron_transport.transport.generate_multiband_eq]
returns the model as a function.

```python
--8<-- "examples/how_to/fitting.py:calculate"
```

## Two-band fit (older API)

[`fit_twoband`][medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor.fit_twoband]
and `calculate_twoband` still exist and fit exactly two bands to one column, with a second
dataset added through `extension`. They take `p0 = [n1, n2, mu1, mu2]`, **not**
interleaved. `fit_multiband` does the same and more, so prefer it for new code;
[tutorial 5](../tutorials.md#5-realistic-workflow-5_realistic_examplepy) shows `fit_twoband`
in full. On plain arrays:
[`etr.fit_twoband`][medapy.analysis.electron_transport.transport.fit_twoband].
