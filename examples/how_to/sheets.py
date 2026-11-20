# --8<-- [start:setup]
from pathlib import Path

import pandas as pd
import pint_pandas  # needed by df.ms.wu
from medapy import ms_pandas, ureg

data_dir = Path(__file__).parent.parent / 'files'
xx_file = data_dir / 'a_twoband_test_I1-5(10mA)_V20-21_Rxx_Bsweep.csv'
xy_file = data_dir / 'a_twoband_test_I1-5(10mA)_V20-40_Rxy_Bsweep.csv'

# Columns: 'Field (T)', 'Resistance (Ohms)'
xx = pd.read_csv(xx_file)
xx.ms.init_msheet(translations={'Ohms': 'ohm'})
xy = pd.read_csv(xy_file)
xy.ms.init_msheet(translations={'Ohms': 'ohm'})
# --8<-- [end:setup]


# --8<-- [start:read]
df = pd.read_csv(xx_file)
df.ms.init_msheet(
    translations={'Ohms': 'ohm'},  # unit spellings Pint does not know
    patch_rename=True,             # make df.rename(...) keep labels and axes in sync
)
print(df.ms)        # data with each column's unit (U), axis (A) and labels (L)
print(df.ms.units)  # {'Field': 'T', 'Resistance': 'Ω'}
print(df.ms.axes)   # {'x': 'Field', 'y': 'Resistance', 'z': None}
# --8<-- [end:read]


# --8<-- [start:set_unit]
df = xx.copy()
df.ms.set_unit('Resistance', 'mohm')  # values stay as they are, only the unit changes
print(df.ms.get_unit('Resistance'))   # mΩ
# --8<-- [end:set_unit]


# --8<-- [start:convert]
df = xx.copy()
df.ms.convert_unit('Field', 'mT')       # values and unit together
df.ms.convert_ms_units(['T', 'mohm'])   # every column at once, in column order
df.ms.convert_unit('Field', 'Oe', contexts='Gaussian')  # T <-> Oe needs a Pint context
print(df.head(3).ms)
# --8<-- [end:convert]


# --8<-- [start:labels]
df = xx.copy()
df.ms.rename({'Resistance': 'R_xx'})  # renames the column, keeps its unit and axis
df.ms.add_labels({'Field': ['B', 'H'], 'R_xx': 'R'})
print(df.ms['B'].head(3))             # a label works wherever a column name does
print(df.ms.get_unit('R'))
# --8<-- [end:labels]


# --8<-- [start:axes]
df = xx.copy()
df.ms.set_as_x('Resistance', swap=True)  # Resistance becomes x, Field takes its old axis (y)
print(df.ms.axes)
df.ms.set_as_axis('Field', 'b')          # custom axis names work too
print(df.ms.x.head(3), df.ms.b.head(3))  # axis data by attribute
# --8<-- [end:axes]


# --8<-- [start:concat]
xx_part, xy_part = xx.copy(), xy.copy()
xx_part.ms.rename({'Resistance': 'Rxx'})  # column names must not clash
xy_part.ms.rename({'Resistance': 'Rxy'})
data = xx_part.ms.concat(xy_part)  # the x column of xy_part is dropped
print(data.head(3).ms)             # Field, Rxx, Rxy
# --8<-- [end:concat]


# --8<-- [start:wu]
df = xx.copy()
current = 10 * ureg.mA                             # I1-5(10mA) in the filename
df.ms['Voltage'] = df.ms.wu('Resistance') * current
df.ms.convert_unit('Voltage', 'mV')                # stored as Ω·mA until converted
df.ms['Conductance'] = 1 / df.ms.wu('Resistance')  # 1/Ω
df.ms.convert_unit('Conductance', 'mS')
print(df.head(3).ms)
# --8<-- [end:wu]


# --8<-- [start:r2rho]
from medapy.analysis import etr  # registers df.etr; needs the fit extra

df = xx.copy()
df.etr.r2rho('xx', col='Resistance', t=400e-9, width=40e-6, length=20e-6,
             new_col='rho_xx', inplace=True)   # floats are in meters
df.ms.convert_unit('rho_xx', 'uohm*cm')

hall = xy.copy()
hall.etr.r2rho('xy', col='Resistance', t=400 * ureg.nm,
               new_col='rho_xy', inplace=True)  # quantities can use any length unit
print(df.head(3).ms, hall.head(3).ms, sep='\n')
# --8<-- [end:r2rho]


# --8<-- [start:save]
out_file = data_dir / 'results' / 'how_to' / 'xx_sheet.csv'
xx.ms.save_msheet(out_file, formatter={'Resistance': '{:.3f}'})  # others: float_format='%.4f'
# header line: Field [T],Resistance [Ω]

back = pd.read_csv(out_file)
back.ms.init_msheet()  # units come back from the header; labels and axes do not
print(back.ms.units)
# --8<-- [end:save]
