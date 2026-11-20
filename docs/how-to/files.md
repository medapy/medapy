# Organize data files

A `MeasurementCollection` reads parameters out of filenames, so you can select, sort and
rename files without opening them. All recipes on this page start from:

```python
--8<-- "examples/how_to/files.py:setup"
```

The folder `examples/files` holds 52 empty `sample_*.csv` files named like
`sample_I1-5(0.5uA)_V2-3_V2-6_Rxx_sweepField_B-14to14T_T=10.0K_Pos=0.0Deg.csv`, plus two
`a_twoband_*` data files.

## Open a folder as a collection

[`MeasurementCollection`][medapy.MeasurementCollection] parses every file in the folder that
matches the glob `file_pattern` (default `'*.*'`). If your filenames are
split into parts by something other than `_`, pass `separator=`. Choose it with care: `-`
also joins contact numbers (`I1-5`) and `=` joins a parameter to its value (`T=10K`), so a
separator containing either splits those parts apart and they are no longer parsed.

```python
--8<-- "examples/how_to/files.py:inspect"
```

## Use your own parameter definitions

A file is parsed only for the parameters it is given. Define new ones with
[`ParameterDefinition`][medapy.ParameterDefinition] and use them alone or together with the
defaults. Pass the same list to `MeasurementCollection`. To keep definitions in a JSON file
instead, see [Parameter definitions](../explanation.md#parameter-definitions).

```python
--8<-- "examples/how_to/files.py:definitions"
```

## Select files by a fixed parameter

Pass the parameter's `name_id` as a keyword to
[`filter`][medapy.MeasurementCollection.filter]. A single value matches exactly; a tuple
`(min, max)` is a range with both bounds included, and `None` leaves a side open. A list
of two values is a range as well, not a set of choices; to get "either value", combine two
filtered collections with `+`.

```python
--8<-- "examples/how_to/files.py:fixed"
```

Files that do not have the parameter at all, or have it swept, never match.

## Select files by sweep and direction

`sweeps=` names parameters that must be swept, and `sweep_directions=` adds the direction.
To select by the swept range, append `_sweep` to the parameter name.

```python
--8<-- "examples/how_to/files.py:sweeps"
```

## Select files by contacts

`contacts=` takes a pair as a tuple, a string like `'I1-5(1uA)'`, a
[`ContactPair`][medapy.ContactPair], or a list of these that must all be present. A tuple
matches the pair whatever its polarization; a string or `ContactPair` with a polarization
and magnitude matches only that. A single integer, such as `contacts=4`, matches a
single-contact entry like `V4`, for files where an individual potential is measured; it
does not match pairs that include contact 4.

```python
--8<-- "examples/how_to/files.py:contacts"
```

## Select by text in the name

`name_contains=` takes a string or a regular expression, or a list that must all match.

```python
--8<-- "examples/how_to/files.py:name"
```

## Combine criteria

`filter` keeps files that match all criteria. For files that match at least one, use
[`filter_generator`][medapy.MeasurementCollection.filter_generator] with `mode='any'`, and
turn the result into a collection by passing it to `MeasurementCollection` as a list.

```python
--8<-- "examples/how_to/files.py:combine"
```

## Exclude files

[`exclude`][medapy.MeasurementCollection.exclude] removes every file that matches *any* of
its criteria. To remove only the files that match *all* of them, use
`filter(..., exclude=True)`.

```python
--8<-- "examples/how_to/files.py:exclude"
```

## Work through a large folder

`filter_generator` checks files only as you iterate, so you can stop early without checking
the whole collection.

```python
--8<-- "examples/how_to/files.py:generator"
```

## Sort

[`sort`][medapy.MeasurementCollection.sort] returns a new collection ordered by one or more
fixed parameters. Every file must have a fixed value for them, so filter first: a file
without the parameter, or with it swept, has nothing to sort by.

```python
--8<-- "examples/how_to/files.py:sort"
```

## Read the parameters of a file

A [`MeasurementFile`][medapy.MeasurementFile] can be made from a name alone; the file does
not need to exist. [`state_of`][medapy.MeasurementFile.state_of] returns a
[`ParameterState`][medapy.collection.parameter.ParameterState] with everything parsed for
one parameter.

```python
--8<-- "examples/how_to/files.py:parameters"
```

## Rename to a naming scheme

[`rename`][medapy.MeasurementFile.rename] changes only the path stored in the object: the
file on disk is not touched. The collection and its subcollections share the same
`MeasurementFile` objects, so rename a `copy()` unless you mean to change the collection.
[`generate_filename`][medapy.MeasurementFile.generate_filename] builds a name from the
parsed contacts and parameters only, so other parts of the old name (`sample`, `Rxx`) are
not in it.

```python
--8<-- "examples/how_to/files.py:rename"
```

## Copy a selection of files to another folder

medapy does not copy, move or delete files. Compute the new paths with `rename` and copy
with `shutil`:

```python
--8<-- "examples/how_to/files.py:copy"
```
