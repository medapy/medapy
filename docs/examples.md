# Examples

This page provides practical examples demonstrating MeDaPy's core functionality. Each example builds on the previous ones, starting with basic concepts and progressing to realistic analysis workflows.

## Example Files

All examples are located in the `/examples` directory with corresponding data files in `/examples/files`.

---

## 1. Measurement Sheet Basics (`1_ms_pandas.py`)

This example introduces the measurement sheet (`ms`) pandas accessor, which provides unit-aware DataFrame functionality.

### Key Concepts

**Initialization**
```python
import pandas as pd
from medapy import ms_pandas

# Create a DataFrame with units in column names
df = pd.DataFrame({
    'Field (Oe)': [1, 2, 3],
    'Current (uA)': [10, 10, 10],
    'Voltage (mV)': [0.05, 0.1, 0.15],
    'Resistance (Ohm)': [5, 10, 15]
})

# Initialize as measurement sheet
custom_unit_dict = dict(Ohm='ohm')  # Map non-standard unit names
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
```

**Labels and Axes**
```python
# Add labels (multiple labels per column allowed)
df.ms.add_labels({
    'Field': ['H', 'B'],
    'Resistance': 'R',
    'Voltage': 'V'
})

# Assign axes (x, y, z, or custom)
df.ms.set_as_axis('R', 'u')           # Add new axis
df.ms.set_as_axis('rho', 'y')          # Reassign y axis
df.ms.set_as_axis('Voltage', 'x', swap=True)  # Swap with existing
```

**Data Access**
```python
# Multiple ways to access data
fld = df['Field']        # Standard pandas
fld = df.ms['Field']     # By column name
fld = df.ms["H"]         # By label
rho = df.ms.y            # By axis
```

**Unit Conversions**
```python
# Method 1: Manual conversion
df.ms['V'] = df.ms['V'] * 1000
df.ms.set_unit('V', 'uV')

# Method 2: Automatic conversion (preferred)
df.ms.convert_unit('H', 'T', contexts='Gaussian')
```

**Unit-Aware Calculations**
```python
# Get columns with units for calculations
crnt = df.ms.wu('I')  # 'wu' = 'with units'
volt = df.ms.wu('V')
df.ms['Conductance'] = crnt / volt  # Units preserved automatically
```

---

## 2. Measurement Collections (`2_collection.py`)

This example demonstrates how to manage collections of measurement files with automatic parameter extraction from filenames.

### Key Concepts

**Parameter Definitions**
```python
from medapy.collection import ParameterDefinition

# Define parameter with names and units
field_param = ParameterDefinition(
    name_id='field',
    long_names=['field', 'Field'],
    short_names=['B', 'H'],
    units=['T', 'Oe', 'G', 'mT']
)
```

**File Management**
```python
from medapy.collection import MeasurementFile

# Create file with parameter parsing
testfile = MeasurementFile(
    "sample_V1-5(1e-3V)_sweepField_B-14to14T_T=3.0K.csv",
    parameters=[field_param, temp_param]
)

# Rename/reorganize files
testfile.rename(
    directory='~',
    name='newname',
    prefix='sample',
    postfix='date',
    sep='_',
    ext='dat'
)

# Merge files
merged = testfile.merge(testfile2, strict_mode=True)
```

**Collections**
```python
from medapy.collection import MeasurementCollection, DefinitionsLoader
from pathlib import Path

# Load default parameter definitions
parameters = DefinitionsLoader().get_all()

# Initialize collection from directory
path = Path('examples/files')
collection = MeasurementCollection(path, parameters)

# Explore collection
collection.head(6)  # Show first 6 files
collection.tail()   # Show last 5 files
```

**Filtering**
```python
from medapy.collection import ContactPair

# Filter by sweep direction
files_down = collection.filter(sweep_direction='dec')

# Filter by temperature range
files_T2_7 = collection.filter_generator(temperature=(2, 7))

# Filter by contact pair
files_V2_6 = collection.filter(contacts=(2, 6))

# Filter by specific contact configuration
pair = ContactPair(1, 5, 'I', 1e-6)  # I1-5(1uA)
files_1uA = collection.filter(contacts=pair)

# Combine multiple filters
files_specific = collection.filter_generator(
    contacts=[pair, (3, 7)],
    polarization='current',
    sweep_direction='inc',
    temperature=(2, 10),
    position=[45, 'IP']
)
```

**Parameter State Access**
```python
# Get parameter state from a file
state = meas_f.state_of('magnetic_field')

# Access parsed values
print(state.is_swept)         # Boolean
print(state.sweep)            # (min, max, direction)
print(state.range)            # (min, max)
print(state.value)            # Single value if not swept
```

---

## 3. Data Processing (`3_proc_pandas.py`)

This example shows common data processing operations using the `proc` accessor.

### Key Concepts

**Monotonicity**
```python
from medapy.analysis import proc_pandas

# Ensure data is monotonically increasing
df.proc.ensure_increasing(inplace=True)
```

**Range Selection**
```python
# Select data in a specific range
df1 = df.proc.select_range((-1, 3), inplace=False)
```

**Normalization**
```python
# Normalize columns
# by='mid' normalizes to midpoint value
# append='norm' adds '_norm' to column name
df.proc.normalize(by='mid', append='norm', inplace=True)
```

**Symmetrization**
```python
# Make data symmetric around zero (for magnetic field sweeps)
df.proc.symmetrize(cols=['R', 'rho', 'V'], inplace=True)
```

**Interpolation**
```python
import numpy as np

# Interpolate to new x-axis values
fld = np.linspace(0, 4, 6)
df2 = df.proc.interpolate(fld, cols=['R', 'rho', 'V'], inplace=False)
```

---

## 4. Electron Transport Analysis (`4_etr_pandas.py`)

This example demonstrates electron transport analysis, including resistance-to-resistivity conversion and Hall fitting.

### Key Concepts

**Basic Setup**
```python
from medapy import ms_pandas, ureg # Global unit registry
from medapy.analysis import electron_transport
import matplotlib.pyplot as plt
```

**Resistance to Resistivity Conversion**
```python
# Define geometry with units
t = 50 * ureg('nm')      # Thickness
w = 10 * ureg('cm')      # Width
l = 2 * ureg('cm')       # Length

# Convert (etr accessor extends proc accessor)
df.etr.r2rho(
    'xx',                # Measurement type
    col='R',             # Input column
    t=t, width=w, length=l,
    new_col='Resistivity',
    add_label='rho',
    inplace=True
)
```

**Linear Hall Fit**
```python
# Fit Hall response (linear in field)
lin_coefs, _ = df.etr.fit_linhall(
    col='rho',
    x_range=(1, None),   # Fit range
    set_axis='l',        # Assign axis to fit column
    add_label='flin',    # Label for fit column
    inplace=True
)
```

**Two-Band Hall Fit**
```python
# Multi-band transport fitting
bands = 'he'  # Hole and electron bands
p0 = [1e26, 1e25, 0.015, 0.02]  # Initial [n1, n2, mu1, mu2] in SI

p_opt, _ = df.etr.fit_twoband(
    p0,
    kind='xx',           # or 'xy' for Hall
    bands=bands,
    report=True,         # Print fit report
    set_axis='f',
    add_label='f2bnd',
    inplace=True
)
```

**Visualization**
```python
fig, ax = plt.subplots()
ax.plot(df.ms.x, df.ms['rho'], 'o', label='data')
ax.plot(df.ms.x, df.ms['flin'], label='linear fit')
ax.plot(df.ms.x, df.ms['f2bnd'], label='two-band fit')
ax.legend()
plt.show()
```

### Method Chaining

Since `etr` extends `proc`, all processing methods are available:
```python
df.etr.ensure_increasing(inplace=True)  # From proc
df.etr.r2rho('xx', ...)                 # From etr
```

---

## 5. Realistic Workflow (`5_realistic_example.py`)

This comprehensive example demonstrates a complete analysis workflow: loading data from collections, processing, fitting, and saving results.

### Workflow Overview

1. **Setup and Data Loading**
```python
from pathlib import Path
import pandas as pd
from medapy import ms_pandas
from medapy.collection import MeasurementCollection, ContactPair, DefinitionsLoader

# Setup directories
data_dir = Path('examples/files')
result_dir = data_dir / 'results'
result_dir.mkdir(exist_ok=True)

# Load parameters and create collection
parameters = DefinitionsLoader().get_all()
collection = MeasurementCollection(data_dir, parameters)
```

2. **Filter Specific Files**
```python
# Filter by contact configuration
pair_10mA = ContactPair(1, 5, 'I', 10e-3)

files_xx = collection.filter(contacts=[pair_10mA, (20, 21)])
files_xy = collection.filter(contacts=[pair_10mA, (20, 40)])

xx_datafile = files_xx.files[0]
xy_datafile = files_xy.files[0]
```

3. **Read and Combine Data**
```python
# Read CSV files
xx = pd.read_csv(xx_datafile.path, delimiter=',', usecols=[0, 1])
xy = pd.read_csv(xy_datafile.path, delimiter=',')

# Initialize measurement sheets
custom_unit_dict = dict(Ohms='ohm')
xx.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)
xy.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

# Rename for clarity
xx.ms.rename({'Resistance': 'Resistance_xx'})
xy.ms.rename({'Resistance': 'Resistance_xy'})

# Concatenate preserving metadata
data = xx.ms.concat(xy)
data.ms.add_labels({
    'Field': 'H',
    'Resistance_xx': 'Rxx',
    'Resistance_xy': 'Rxy'
})
```

4. **Process and Convert**
```python
# Ensure monotonicity
data.etr.ensure_increasing(inplace=True)

# Define geometry (can use values or pint quantities)
length = 20e-6  # meters
width = 40e-6
t = 400e-9

# Convert to resistivity
data.etr.r2rho('xx', col='Rxx', t=t, width=width, length=length,
               new_col='Resistivity_xx', add_label='rho_xx',
               inplace=True)
data.etr.r2rho('xy', col='Rxy', t=t,
               new_col='Resistivity_xy', add_label='rho_xy',
               inplace=True)
```

5. **Fitting**
```python
# Linear Hall fit
lin_coefs, _ = data.etr.fit_linhall(
    x_range=(11, None),
    add_label='flin',
    inplace=True
)

# Select data range for xx fitting
part_xx = data.ms[['Field', 'rho_xx']]
part_xx = part_xx.etr.select_range((-6, 6), inside_range=False)

# Two-band fitting with extension
bands = 'he'
p0 = [1e26, 1e25, 0.015, 0.02]

p_opt_xy, _ = data.etr.fit_twoband(
    p0, col='rho_xy', kind='xy', bands=bands,
    extension=part_xx,  # Extend fit to additional data
    add_label='f2_xy',
    report=True,
    inplace=True
)

p_opt_xx, _ = data.etr.fit_twoband(
    p0, col='rho_xx', kind='xx', bands=bands,
    field_range=(-6, 6), inside_range=False,
    extension=(data.ms.x, data.ms.y),
    add_label='f2_xx',
    report=True,
    inplace=True
)
```

6. **Save Results**
```python
# Format output (scientific notation for specific columns)
cols = ['rho_xx', 'rho_xy', 'flin', 'f2_xy', 'f2_xx']
fmtr = {col: '{:.4E}' for col in cols}

data.ms.save_msheet(result_dir / 'data.csv', formatter=fmtr)
```

7. **Visualization**
```python
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(10, 5))

# Plot xx data
ax1.plot(data.ms.x, data.ms['rho_xx'], '.', label='data')
ax1.plot(data.ms.x, data.ms['f2_xx'], 'r--', label='fit')

# Plot xy data
ax2.plot(data.ms.x, data.ms['rho_xy'], '.', label='data')
ax2.plot(data.ms.x, data.ms['f2_xy'], 'r--', label='fit')
ax2.plot(data.ms.x, data.ms['flin'], 'k--', label='linear fit')

plt.show()
```

---

## Common Patterns

### Unit Registry

Always use the package's global unit registry:
```python
from medapy import ureg
```

### Inplace Operations

Most methods support `inplace` parameter (mimicking pandas):
```python
df.etr.normalize(inplace=True)   # Modifies df
df2 = df.etr.normalize(inplace=False)  # Returns new DataFrame
```

### File Organization

Use `pathlib.Path` for cross-platform compatibility:
```python
from pathlib import Path
data_dir = Path(__file__).parent / 'data'
```
