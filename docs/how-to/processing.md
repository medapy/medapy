# Process data

The `df.proc` accessor processes a measurement sheet along its x axis. The same methods are
on `df.etr`, and each has a counterpart in
[`medapy.analysis.processing`][medapy.analysis.processing] for plain arrays. All recipes on
this page start from:

```python
--8<-- "examples/how_to/processing.py:setup"
```

`data` holds an xx and a Hall resistance measured at the same 32 field values, from -13.98
to 13.14 T. Every method below returns a new sheet and leaves `data` as it is; pass
`inplace=True` to change the sheet itself. Where a method takes `cols`, it defaults to the y
axis column.

## Make x increasing

Several methods need x to be monotonic, and some, such as `interpolate`, assume it
increases.
[`check_monotonic`][medapy.analysis.proc_pandas.DataProcessingAccessor.check_monotonic]
tells which way it goes, and
[`ensure_increasing`][medapy.analysis.proc_pandas.DataProcessingAccessor.ensure_increasing]
reverses a decreasing sweep.

```python
--8<-- "examples/how_to/processing.py:monotonic"
```

On plain arrays:
[`processing.check_monotonic_arr`][medapy.analysis.processing.check_monotonic_arr] and
[`processing.check_monotonic_df`][medapy.analysis.processing.check_monotonic_df].

## Select or cut out a range

[`select_range`][medapy.analysis.proc_pandas.DataProcessingAccessor.select_range] keeps the
rows with x inside a range, or outside it with `inside_range=False`. It needs monotonic x.
[`filter_range`][medapy.analysis.proc_pandas.DataProcessingAccessor.filter_range] takes the
same arguments and compares every x value instead, so x can be in any order; `select_range`
is faster on sorted data.

```python
--8<-- "examples/how_to/processing.py:range"
```

On plain arrays: [`processing.select_range_arr`][medapy.analysis.processing.select_range_arr]
and [`processing.filter_range_arr`][medapy.analysis.processing.filter_range_arr]
(`select_range_df` and `filter_range_df` for DataFrames).

## Interpolate onto a new grid

[`interpolate`][medapy.analysis.proc_pandas.DataProcessingAccessor.interpolate] puts the
given columns onto new x values. The result holds only the x column and those columns. Keep
the new x inside the measured range: the default linear interpolation does not extrapolate.
Pass `interp=` for another method, a function `f(x, y, x_new)`.

```python
--8<-- "examples/how_to/processing.py:interpolate"
```

On plain arrays: [`processing.interpolate`][medapy.analysis.processing.interpolate].

## Symmetrize a measured sweep

[`symmetrize`][medapy.analysis.proc_pandas.DataProcessingAccessor.symmetrize] keeps the part
of a column that is even in x, and
[`antisymmetrize`][medapy.analysis.proc_pandas.DataProcessingAccessor.antisymmetrize] the odd
part. Both pair row *i* with row *-i*, so they need an x grid symmetric about zero. A
measured grid rarely is (this one is offset by 0.0175 T), so interpolate onto
[`symmetric_range`][medapy.analysis.processing.symmetric_range] first.

```python
--8<-- "examples/how_to/processing.py:symmetrize"
```

On plain arrays: [`processing.symmetrize`][medapy.analysis.processing.symmetrize] and
[`processing.antisymmetrize`][medapy.analysis.processing.antisymmetrize].

## Smooth

[`savgol_filter`][medapy.analysis.proc_pandas.DataProcessingAccessor.savgol_filter] fits a
polynomial in a moving window (it needs the `fit` extra, for scipy);
[`moving_average`][medapy.analysis.proc_pandas.DataProcessingAccessor.moving_average]
averages over the window. Even windows are widened by one. By default, the points at each
end that lack a full window are set to NaN; `edge_mode='drop'` removes those rows instead, a
`numpy.pad` mode such as `'edge'` or `'reflect'` pads the data before smoothing, and
`savgol_filter` also accepts `'interp'`, which fits the edges with the polynomial.

```python
--8<-- "examples/how_to/processing.py:smooth"
```

On plain arrays: [`processing.savgol_filter`][medapy.analysis.processing.savgol_filter] and
[`processing.moving_average`][medapy.analysis.processing.moving_average].

## Normalize

[`normalize`][medapy.analysis.proc_pandas.DataProcessingAccessor.normalize] divides columns
by a number or by their value in the `'first'`, `'mid'` or `'last'` row. `'mid'` is the
middle row, which is at zero field only on a symmetric grid. The result is dimensionless.

```python
--8<-- "examples/how_to/processing.py:normalize"
```

On plain arrays: [`processing.normalize`][medapy.analysis.processing.normalize].
