# Work with measurement sheets

A measurement sheet is a pandas `DataFrame` that knows the unit of each column, plus labels
and axes for convenient access, through the `df.ms` accessor. All recipes on this page start
from:

```python
--8<-- "examples/how_to/sheets.py:setup"
```

The two files hold a longitudinal (`xx`) and a Hall (`xy`) resistance measured against
field, 32 points each. Recipes that change a sheet work on a `copy()`.

## Read a file into a sheet

Read the file with pandas, then call
[`init_msheet`][medapy.ms_pandas.MeasurementSheetAccessor.init_msheet]: it takes the units
from the column names (`Field (T)`), sets the first three columns as the x, y and z axes, and
strips the units from the names.

```python
--8<-- "examples/how_to/sheets.py:read"
```

## Fix a unit the file states wrongly

If the column name has no unit, or the wrong one, set it with
[`set_unit`][medapy.ms_pandas.MeasurementSheetAccessor.set_unit]. It does not touch the
values. If the file writes a unit in a way Pint does not understand (`Ohms` here), map it
with `translations=` in `init_msheet` instead.

```python
--8<-- "examples/how_to/sheets.py:set_unit"
```

## Convert units

[`convert_unit`][medapy.ms_pandas.MeasurementSheetAccessor.convert_unit] changes values and
unit together; `convert_ms_units` does every column in one call.

```python
--8<-- "examples/how_to/sheets.py:convert"
```

## Give columns short names

[`rename`][medapy.ms_pandas.MeasurementSheetAccessor.rename] renames columns and keeps their
units, axes and labels. [`add_labels`][medapy.ms_pandas.MeasurementSheetAccessor.add_labels]
adds aliases: a column can have several, and `df.ms` methods accept a label wherever they
take a column name.

```python
--8<-- "examples/how_to/sheets.py:labels"
```

## Choose the axes

Processing and fitting methods act on the x axis, and on the y axis unless told otherwise.
[`set_as_x`][medapy.ms_pandas.MeasurementSheetAccessor.set_as_x] (and `set_as_y`,
`set_as_z`, `set_as_axis`) reassign them.

```python
--8<-- "examples/how_to/sheets.py:axes"
```

## Combine xx and xy files into one sheet

[`concat`][medapy.ms_pandas.MeasurementSheetAccessor.concat] joins sheets side by side and
merges their units and labels. Rename columns first so the names do not clash. By default it
drops the x column of the other sheets, which assumes they were measured on the same x
values; it does not check this. If they were not, interpolate them onto a common grid first
(see [Interpolate onto a new grid](processing.md#interpolate-onto-a-new-grid)).

```python
--8<-- "examples/how_to/sheets.py:concat"
```

## Calculate a new column with units

[`wu`][medapy.ms_pandas.MeasurementSheetAccessor.wu] ("with units") returns a column as a
Pint-typed series. Arithmetic on it carries the units, and assigning the result through
`df.ms[...]` stores the unit with the new column. `wu` needs `pint_pandas` to be imported.

```python
--8<-- "examples/how_to/sheets.py:wu"
```

## Convert resistance to resistivity

[`r2rho`][medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor.r2rho]
lives on the `df.etr` accessor, which needs the `fit` extra
(`pip install "medapy[fit]"`). `'xx'` needs thickness, width and length; `'xy'` only the
thickness. Give them as floats in meters or as Pint quantities; for `'xx'`, width and length
must be both floats or both quantities.

```python
--8<-- "examples/how_to/sheets.py:r2rho"
```

On plain arrays (floats in meters only):
[`etr.r2rho`][medapy.analysis.electron_transport.transport.r2rho].

## Save and read back

[`save_msheet`][medapy.ms_pandas.MeasurementSheetAccessor.save_msheet] writes a CSV with the
units back in the column names; `formatter` sets the number format per column (by name or
label). Reading it with `init_msheet` restores the units.

```python
--8<-- "examples/how_to/sheets.py:save"
```
