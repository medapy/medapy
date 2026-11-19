# Explanation

This page covers the ideas behind MeDaPy's API, the things that are hard to see from the
[Tutorials](tutorials.md) or the [Reference](reference/index.md) alone: where a measurement sheet
keeps its state, how a filename turns into parameters, and how the pieces fit together.

## The measurement sheet model

A *measurement sheet* is an ordinary `pandas.DataFrame` that the `.ms` accessor has
initialized with `df.ms.init_msheet()`. MeDaPy does not subclass `DataFrame`; everything it
knows about a sheet is stored in the DataFrame's own `attrs` dictionary:

| `attrs` key | Accessor property | Holds |
| --- | --- | --- |
| `_ms_units` | `df.ms.units` | column name → unit string |
| `_ms_labels` | `df.ms.labels` | label → column name |
| `_ms_axes` | `df.ms.axes` | axis (`x`, `y`, `z`, or a custom name) → column name |
| `_ms_unit_pattern` | — | regex used to parse units out of column names |
| `_ms_unit_brackets` | — | brackets used when writing units back into names |
| `_ms_translations` | — | non-Pint → Pint unit spellings, e.g. `{'Ohm': 'ohm'}` |

`df.ms.get_ms_state()` returns this dictionary and `df.ms.set_ms_state(state)` writes it onto
another DataFrame, which is how to carry the sheet's metadata over to a DataFrame built from
scratch.

### What survives pandas operations

Because the state lives in `attrs`, it follows the DataFrame wherever pandas propagates
`attrs`: copies, boolean and `.loc` slicing, arithmetic, `reset_index`, and `pd.concat` of
sheets whose `attrs` are identical. Concatenating sheets with *different* `attrs` drops them;
use `df.ms.concat(...)` instead, which merges units, labels and axes and rejects duplicate
column names or labels.

`attrs` is plain data, though: pandas copies it but knows nothing about what it means. When a
column is renamed with `DataFrame.rename`, the units, labels and axes still point to the old
name. Rename columns with `df.ms.rename({...})`, which updates all three maps, or initialize
the sheet with `init_msheet(patch_rename=True)`. That option replaces the sheet's own `rename`
method with one that keeps the maps in sync, so `df.rename(columns={...})` works as expected.

### Column names, labels and axes

These are three ways of naming the same column, each with a different purpose:

- The **column name** is the DataFrame column itself, with the unit stripped:
  `Field (Oe)` becomes `Field` with unit `Oe`.
- A **label** is an alias. A column can have several (`H`, `B` and `Champ` for `Field`), and
  `df.ms['H']` and `df.ms.wu('H')` accept labels and column names alike. Labels are what
  analysis code should use, so a script works no matter what the instrument called a column.
- An **axis** is a role. `x`, `y` and `z` always exist and are set to the first three columns
  on initialization; custom axes can be added. Processing methods default to the `x` and `y`
  axes, so `df.proc.symmetrize()` needs no column arguments. Axes are exposed as attributes:
  `df.ms.x` is the data of the x-axis column, and `df.proc.col_x` its name.

Each column has at most one axis; `set_as_axis(..., swap=True)` swaps two columns' axes.

### Units in column names

`init_msheet` reads a unit from the last bracketed part of each column name: `(...)`, `[...]`
or `{...}`. Columns without one are dimensionless. The unit string must be something Pint
understands; spellings it does not, like `Ohm`, are mapped through `translations`:

```python
df.ms.init_msheet(translations={'Ohm': 'ohm'})
```

When the sheet is written back with `save_msheet`, units are put into the column names
again, in square brackets by default (see `set_unit_brackets`).

Stripping units can make two names collide, for example `I [mA]` and `I [A]` both become `I`.
`infer_ms_units(handle_duplicates=...)` decides what happens then:

| Option | Result for `I [mA]`, `I [A]` |
| --- | --- |
| `'suffix'` (default) | `I_mA`, `I_A` |
| `'preserve'` | `I [mA]`, `I [A]` (names kept as they are) |
| `'mangle'` | `I`, `I.1` (pandas style) |
| `'error'` | raises `ValueError` |

Only colliding names are affected; the others are stripped as usual.

## Units and the registry

Pint quantities can only be combined if they come from the same `UnitRegistry`. MeDaPy
creates one registry when it is imported and installs it as Pint's *application registry*, so
every part of the package uses it. Import it from the package root rather than creating
your own:

```python
from medapy import ureg
```

Inside a sheet, columns hold plain floats and the unit is stored separately in `df.ms.units`.
This keeps the data compatible with everything that expects NumPy arrays, such as plotting,
SciPy and the `.proc` methods. When units are needed in a calculation, `df.ms.wu('R')`
("with units") returns the column as a Pint-typed Series. Arithmetic on such Series carries
the units, and assigning the result through `df.ms[...]` stores the computed unit:

```python
import pint_pandas  # required for Pint-typed Series

df.ms['G'] = 1 / df.ms.wu('R')   # unit of G is 1/Ω
```

By contrast, `df.ms['R']` and `df['Resistance']` return plain values. `df.ms['R']` puts the
unit into the Series name (`Resistance [Ω]`), which makes it convenient for plot labels.

`convert_unit('H', 'T')` rescales values and updates the stored unit together. Some
conversions need a Pint context: tesla ↔ oersted, for instance, requires `'Gaussian'`,
enabled on the shared registry with `ureg.enable_contexts('Gaussian')`.

By default (`strict_units=True`), assigning a Pint Series to an existing column with a
different unit raises an error instead of silently overwriting the unit.

## The accessor family

MeDaPy extends pandas through accessors, which are registered when their module is imported:

| Import | Accessor | Purpose |
| --- | --- | --- |
| `from medapy import ms_pandas` | `df.ms` | units, labels, axes, saving |
| `from medapy.analysis import proc_pandas` | `df.proc` | generic processing: range selection, symmetrization, normalization, interpolation, smoothing |
| `from medapy.analysis import etr` | `df.etr` | electron transport: resistivity, Hall and multiband fits; includes everything on `.proc` |

The accessor classes are never instantiated by hand. `.proc` and `.etr` require a sheet
that has already been initialized with `.ms` (they read its axes), and `.etr` needs the
`fit` extra.

Most `.proc`/`.etr` methods follow the pandas convention of returning a new DataFrame unless
`inplace=True` is given. Methods that compute a new quantity, like `normalize` or a fit, can
also append it as a new column (`append=...` or `add_col=...`) instead of overwriting.

### Functional API

Every accessor method is a thin wrapper around a function that works on plain arrays.
`medapy.analysis.processing` holds the functions behind `.proc`; the electron transport
functions are available as `etr` itself:

```python
from medapy.analysis import etr

etr.fit_twoband(field, rho, p0, kind='xy', bands='he')   # arrays in, fit results out
df.etr.fit_twoband(p0, kind='xy', bands='he')            # same fit on the sheet's axes
```

Use the accessor when working with sheets: it picks the columns from the axes, handles
units and writes results back as columns. Use the functions for data that is not in a sheet,
or inside your own fitting code.

!!! note
    The parameter order differs between the two fit families: `fit_twoband` takes
    `p0 = [n1, n2, mu1, mu2]`, while `fit_multiband` takes them interleaved,
    `[n1, mu1, n2, mu2, ...]`.

## Filenames as metadata

It is often convenient to store the conditions of a run (temperature, field range,
contacts) in the filename rather than inside the data file: the data stays a plain table
that any program can open, and the metadata is visible in every file browser.
`MeasurementFile` parses those names, and `MeasurementCollection` lets you filter a folder
of files by the parsed values.

Take one of the example files:

```text
sample_I1-5(0.5uA)_V2-3_V2-6_Rxx_sweepField_B-14to14T_T=10.0K_Pos=0.0Deg.csv
```

The name (without extension) is split at the separator (`_` by default) and each part is
tried in turn:

| Part | Interpreted as |
| --- | --- |
| `sample`, `Rxx` | nothing; free text, still searchable with `name_contains` |
| `I1-5(0.5uA)` | contact pair 1–5, current polarization, magnitude 0.5 µA |
| `V2-3`, `V2-6` | contact pairs 2–3 and 2–6, voltage polarization |
| `sweepField` | `magnetic_field` is swept |
| `B-14to14T` | `magnetic_field` range −14 to 14 T, sweep direction increasing |
| `T=10.0K` | `temperature` fixed at 10 K |
| `Pos=0.0Deg` | `position` fixed at 0° |

### Contact pairs

A contact part has the form `<I|V><first>[-<second>][(<magnitude><A|V>)]`: `I1-5(10mA)`,
`V2-3`, or a single contact `V4`. The letter is the *polarization*: `I` for a pair
carrying current, `V` for one measuring voltage. The magnitude is optional and may use any
SI prefix; one that cannot be parsed raises a `MagnitudeFormatWarning` and the pair is kept
without it.

### Fixed, range and sweep parameters

Each parameter can appear in one of three shapes:

- **Fixed:** a name, an optional `=`, a value and an optional unit: `T=10.0K`, `T3K`.
- **Range:** `<name><start>to<end><unit>`: `B-14to14T`. The parameter is swept, and the order
  of start and end gives the **sweep direction**: `B-14to14T` is increasing, `B14to-14T`
  decreasing. The state stores `min` and `max` in sorted order plus the direction.
- **Sweep:** `sweep<name>` or `<name>sweep`: `sweepField`. The parameter is swept but the
  range is unknown, unless a range part for it also appears.

Units may carry SI prefixes (`300mK` is stored as 0.3 K). The parsed result is available as a
read-only `ParameterState` via `mfile.state_of('temperature')`, or more directly with
`value_of` and `range_of`.

Filters mirror these shapes: a single value matches a fixed parameter exactly, a tuple
`(min, max)` matches a fixed value within the range, `sweeps=` and `sweep_directions=` select
swept parameters, and `contacts=` accepts a contact number, a pair tuple, a string like
`'I1-5(1uA)'` or a `ContactPair`.

The mapping also works in reverse: `generate_filename()` builds a filename from the current
parameters and contacts, so a file derived from a measurement (a merged or processed one)
can be named consistently.

## Parameter definitions

Which names count as which parameter is not hard-coded. Each parameter is described by a
`ParameterDefinition`, and `DefinitionsLoader` reads definitions from JSON. The defaults
ship with the package:

```json
--8<-- "src/medapy/collection/parameter_definitions.json"
```

The fields are:

| Field | Meaning |
| --- | --- |
| key (`"temperature"`) | the parameter's `name_id`, used in `state_of`, filters, etc. |
| `long_names`, `short_names` | names that identify the parameter in a filename; they must not overlap |
| `units` | accepted units; prefixed forms (`mK`, `uT`) are recognized automatically |
| `special_values` | words that stand for a number, e.g. `IP` → 90 for the in-plane position |
| `si_formatting` | write values with SI prefixes when generating filenames |
| `precision` | digits when generating filenames; `"auto"` drops unnecessary trailing zeros |
| `patterns` (optional) | override the regexes for `fixed`, `range` or `sweep` |

Custom patterns are templates with the placeholders `{NAME}` (long or short names),
`{LNAME}`, `{SNAME}`, `{VALUE}` and `{UNIT}`. The defaults are:

| Shape | Default pattern |
| --- | --- |
| `fixed` | `{NAME}=?{VALUE}{UNIT}?$` |
| `range` | `{NAME}{VALUE}to{VALUE}{UNIT}?$` |
| `sweep` | `sweep{NAME}` or `{NAME}sweep` |

To add or change parameters, either pass your own JSON file to
`DefinitionsLoader('my_definitions.json')` (it is loaded on top of the defaults, so
entries with the same key replace them), or create `ParameterDefinition` objects in code, as
in the [collection tutorial](tutorials.md#2-measurement-collections-2_collectionpy). A
`MeasurementCollection` or `MeasurementFile` parses only the parameters it is given, so
passing a subset of definitions is also a way to ignore the rest.
