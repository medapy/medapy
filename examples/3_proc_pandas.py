# --8<-- [start:setup]
import pandas as pd
import numpy as np
from medapy import ms_pandas
from medapy.analysis import proc_pandas


# Create sample DataFrame
df = pd.DataFrame({'Field (Oe)': [5, 3, 1, 0, -1, -3, -5],
                    'Current (uA)': [10, 10, 10, 10, 10, 10, 10],
                    'Voltage (mV)': [103, 49, 22, 10, 18, 51, 97],
                    'Resistance (Ohm)': [10300,  4900,  2200, 1000,  1800,  5100,  9700],
                    'Resistivity (Ohm*cm)': [0.2575, 0.1225, 0.055, 0.025, 0.045, 0.1275, 0.2425],
                    })

custom_unit_dict = dict(Ohm='ohm')
df.ms.init_msheet(translations=custom_unit_dict, patch_rename=True)

# Add labels
df.ms.add_labels({'Field': 'H', 'Resistance': 'R',
                  'Voltage': 'V', 'Current': 'I',
                  'Resistivity': 'rho'})

# Set y axis
df.ms.set_as_y('R')
# --8<-- [end:setup]

# Display current DataFrame
print('\nOriginal MSheet:')
print(df.ms)


# Functionality
# --8<-- [start:monotonic]
# is_monotonic = df.proc.check_monotonic() # check if monotonic manually
df.proc.ensure_increasing(inplace=True)
# --8<-- [end:monotonic]
print('\nMSheet forced to monotonic increasing:')
print(df.ms)

# --8<-- [start:select]
# Select range
df1 = df.proc.select_range((-1, 3), inplace=False)
# --8<-- [end:select]
print('\nMSheet in range:')
print(df1.ms)

# --8<-- [start:normalize]
# Normalizing
# calling functions with cols=None defaults to y-axis column
df.proc.normalize(by='mid', append='norm', inplace=True)
# --8<-- [end:normalize]
print('\nMSheet after normalizing:')
print(df.ms)

# --8<-- [start:symmetrize]
# Symmetrizing
df.proc.symmetrize(cols=['R', 'rho', 'V'], inplace=True)
# --8<-- [end:symmetrize]
print('\nMSheet after symmetrization:')
print(df.ms)

# --8<-- [start:interpolate]
# Interpolation
fld = np.linspace(0, 4, 6)
df2 = df.proc.interpolate(fld, cols=['R', 'rho', 'V'], inplace=False)
# --8<-- [end:interpolate]
print('\nMSheet with interpolated values:')
print(df2.ms)
