# Tutorials

This page walks through MeDaPy's core functionality. Each section builds on the previous ones, starting with basic concepts and progressing to a realistic analysis workflow.

## Example Files

Each section follows one script in the [`examples`](https://github.com/medapy/medapy/tree/master/examples) directory of the repository, with data files in `examples/files`. The code on this page is included directly from those scripts, and the scripts are run by the test suite, so what you see here is code that works. Sections 4 and 5 need the `fit` and `plot` extras.

---

## 1. Measurement Sheet Basics (`1_ms_pandas.py`)

This example introduces the measurement sheet (`ms`) pandas accessor, which provides unit-aware DataFrame functionality.

### Key Concepts

**Initialization**

Units are parsed from the column names. `translations` maps unit spellings that Pint does not understand to ones it does.

```python
--8<-- "examples/1_ms_pandas.py:init"
```

**Labels and Axes**

Labels are aliases for columns; a column can have several. Axes (`x`, `y`, `z`, or any custom name) point at columns and can be reassigned or swapped.

```python
--8<-- "examples/1_ms_pandas.py:labels_axes"
```

**Data Access**

Columns can be reached by name, by label, or by axis.

```python
--8<-- "examples/1_ms_pandas.py:access"
```

**Unit Conversions**

Either change values and unit by hand, or let `convert_unit` do both.

```python
--8<-- "examples/1_ms_pandas.py:units"
```

**Unit-Aware Calculations**

`wu` ("with units") returns a column as a Pint-typed Series, so arithmetic carries units into the new column.

```python
--8<-- "examples/1_ms_pandas.py:wu"
```

---

## 2. Measurement Collections (`2_collection.py`)

This example demonstrates how to manage collections of measurement files with automatic parameter extraction from filenames.

### Key Concepts

**Imports**

```python
--8<-- "examples/2_collection.py:imports"
```

**Parameter Definitions**

A `ParameterDefinition` tells the parser which names and units identify a parameter in a filename.

```python
--8<-- "examples/2_collection.py:definitions"
```

**File Management**

A `MeasurementFile` parses its filename into parameters and contact pairs. Its path can be changed without touching the parsed state, two files can be merged, and a filename can be generated back from the state.

```python
--8<-- "examples/2_collection.py:file"
```

**Collections**

`DefinitionsLoader` provides the default parameter definitions. A `MeasurementCollection` built from a folder parses every file in it.

```python
--8<-- "examples/2_collection.py:collection"
```

Collections behave much like lists:

```python
--8<-- "examples/2_collection.py:list_ops"
```

**Parameter State Access**

```python
--8<-- "examples/2_collection.py:state"
```

**Filtering**

`filter` returns a new collection; `filter_generator` yields matching files lazily. Criteria can be combined.

```python
--8<-- "examples/2_collection.py:filter"
```

---

## 3. Data Processing (`3_proc_pandas.py`)

This example shows common data processing operations using the `proc` accessor.

### Key Concepts

**Setup**

```python
--8<-- "examples/3_proc_pandas.py:setup"
```

**Monotonicity**

```python
--8<-- "examples/3_proc_pandas.py:monotonic"
```

**Range Selection**

```python
--8<-- "examples/3_proc_pandas.py:select"
```

**Normalization**

`by='mid'` normalizes to the value at the midpoint; `append='norm'` writes the result to new columns with a `_norm` suffix instead of overwriting.

```python
--8<-- "examples/3_proc_pandas.py:normalize"
```

**Symmetrization**

Makes data symmetric around zero, as for magnetic field sweeps.

```python
--8<-- "examples/3_proc_pandas.py:symmetrize"
```

**Interpolation**

```python
--8<-- "examples/3_proc_pandas.py:interpolate"
```

---

## 4. Electron Transport Analysis (`4_etr_pandas.py`)

This example demonstrates electron transport analysis, including resistance-to-resistivity conversion and Hall fitting.

### Key Concepts

**Basic Setup**

```python
--8<-- "examples/4_etr_pandas.py:setup"
```

**Processing Methods**

Since `etr` extends `proc`, all processing methods are available on it too.

```python
--8<-- "examples/4_etr_pandas.py:proc_methods"
```

**Resistance to Resistivity Conversion**

Geometry can be given as Pint quantities (as here) or as plain floats in meters.

```python
--8<-- "examples/4_etr_pandas.py:r2rho"
```

**Linear Hall Fit**

```python
--8<-- "examples/4_etr_pandas.py:linhall"
```

**Two-Band Fit**

```python
--8<-- "examples/4_etr_pandas.py:twoband"
```

**Visualization**

```python
--8<-- "examples/4_etr_pandas.py:plot"
```

---

## 5. Realistic Workflow (`5_realistic_example.py`)

This comprehensive example demonstrates a complete analysis workflow: loading data from collections, processing, fitting, and saving results.

### Workflow Overview

1. **Setup and Data Loading**

    ```python
    --8<-- "examples/5_realistic_example.py:load"
    ```

2. **Filter Specific Files**

    ```python
    --8<-- "examples/5_realistic_example.py:filter"
    ```

3. **Read and Combine Data**

    ```python
    --8<-- "examples/5_realistic_example.py:read"
    ```

4. **Process and Convert**

    ```python
    --8<-- "examples/5_realistic_example.py:resistivity"
    ```

5. **Fitting**

    The xy fit uses `extension` to fit the xx data self-consistently with the same band parameters.

    ```python
    --8<-- "examples/5_realistic_example.py:fit"
    ```

6. **Save Results**

    ```python
    --8<-- "examples/5_realistic_example.py:save"
    ```

7. **Visualization**

    ```python
    --8<-- "examples/5_realistic_example.py:plot"
    ```

---

## Next steps

- [How-to guides](how-to/index.md): short recipes for specific tasks, such as selecting files,
  symmetrizing a sweep or constraining a multiband fit.
- [Explanation](explanation.md): how sheets keep their state, how units work, and how
  filenames are parsed.
