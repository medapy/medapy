# --8<-- [start:setup]
from pathlib import Path

import numpy as np
import pandas as pd
from medapy import ms_pandas
from medapy.analysis import proc_pandas, processing

data_dir = Path(__file__).parent.parent / 'files'
xx = pd.read_csv(data_dir / 'a_twoband_test_I1-5(10mA)_V20-21_Rxx_Bsweep.csv')
xy = pd.read_csv(data_dir / 'a_twoband_test_I1-5(10mA)_V20-40_Rxy_Bsweep.csv')
xx.ms.init_msheet(translations={'Ohms': 'ohm'})
xy.ms.init_msheet(translations={'Ohms': 'ohm'})
xx.ms.rename({'Resistance': 'Rxx'})
xy.ms.rename({'Resistance': 'Rxy'})

data = xx.ms.concat(xy)  # Field (x, -13.98..13.14 T), Rxx (y), Rxy
# --8<-- [end:setup]


# --8<-- [start:monotonic]
sweep_down = data.iloc[::-1].reset_index(drop=True)  # same data, field decreasing
print(sweep_down.proc.check_monotonic())  # 1 increasing, -1 decreasing, 0 neither

sweep_up = sweep_down.proc.ensure_increasing()  # reverses the rows; raises if x is not monotonic
print(sweep_up.proc.check_monotonic())    # 1
# --8<-- [end:monotonic]


# --8<-- [start:range]
low_field = data.proc.select_range((-6, 6))                      # -6 T <= B <= 6 T
high_field = data.proc.select_range((-6, 6), inside_range=False)  # the rest
above_10T = data.proc.select_range((10, None))                   # open boundary
no_bounds = data.proc.select_range((-6, 6), inclusive='neither')  # or 'left', 'right'

shuffled = data.sample(frac=1, random_state=0)  # x in no particular order
low_field_too = shuffled.proc.filter_range((-6, 6))
# --8<-- [end:range]
print(len(low_field), len(high_field), len(above_10T), len(no_bounds), len(low_field_too))


# --8<-- [start:interpolate]
field = np.linspace(-12, 12, 97)  # inside the measured field range
on_grid = data.proc.interpolate(field, cols=['Rxx', 'Rxy'])
print(on_grid.head(3).ms)         # Field, Rxx, Rxy on the new grid
# --8<-- [end:interpolate]


# --8<-- [start:symmetrize]
field = processing.symmetric_range(13, dx=0.25)  # -13, -12.75, ..., 12.75, 13
sym = data.proc.interpolate(field, cols=['Rxx', 'Rxy'])
sym = sym.proc.symmetrize('Rxx', append='sym')       # new column Rxx_sym
sym = sym.proc.antisymmetrize('Rxy', append='asym')  # new column Rxy_asym
print(sym.iloc[[0, 52, -1]].ms)                      # B = -13, 0, 13 T
# --8<-- [end:symmetrize]


# --8<-- [start:smooth]
# Savitzky-Golay over 7 points with a 2nd-order polynomial; needs the fit extra (scipy)
smooth = data.proc.savgol_filter(7, cols='Rxx', order=2, append='sg')
# Moving average over 5 points
smooth = smooth.proc.moving_average(5, cols='Rxx', append='avg')
print(smooth.head(4).ms)  # points without a full window are NaN

padded = data.proc.moving_average(5, cols='Rxx', append='avg', edge_mode='edge')
# --8<-- [end:smooth]
print(padded.head(4).ms)


# --8<-- [start:normalize]
norm = data.proc.normalize('Rxx', by='mid', append='norm')  # Rxx / Rxx in the middle row
by_value = data.proc.normalize(['Rxx', 'Rxy'], by=21.8, append='norm')  # or 'first', 'last'
print(norm.iloc[14:18].ms)  # Rxx_norm is dimensionless
# --8<-- [end:normalize]
