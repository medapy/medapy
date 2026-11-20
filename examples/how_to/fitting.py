# --8<-- [start:setup]
from pathlib import Path

import numpy as np
import pandas as pd
from medapy import ms_pandas
from medapy.analysis import etr, processing  # etr needs the fit extra

data_dir = Path(__file__).parent.parent / 'files'
xx = pd.read_csv(data_dir / 'a_twoband_test_I1-5(10mA)_V20-21_Rxx_Bsweep.csv')
xy = pd.read_csv(data_dir / 'a_twoband_test_I1-5(10mA)_V20-40_Rxy_Bsweep.csv')
xx.ms.init_msheet(translations={'Ohms': 'ohm'})
xy.ms.init_msheet(translations={'Ohms': 'ohm'})
xx.ms.rename({'Resistance': 'Rxx'})
xy.ms.rename({'Resistance': 'Rxy'})
data = xx.ms.concat(xy)

# Resistivity in ohm*m; sample size in m
data.etr.r2rho('xx', col='Rxx', t=400e-9, width=40e-6, length=20e-6,
               new_col='rho_xx', inplace=True)
data.etr.r2rho('xy', col='Rxy', t=400e-9, new_col='rho_xy', inplace=True)
# --8<-- [end:setup]


# --8<-- [start:polynomial]
x = np.linspace(-5, 5, 21)
y = 1.5 + 0.3 * x + 0.8 * x**2

coefs = processing.quick_fit(x, y, degrees=2)  # [c0, c1, c2] for x^0, x^1, x^2
curve = processing.make_curve(x, coefs)

line = processing.quick_fit(x, y, degrees=1, x_range=(2, None))  # fit only x >= 2
# --8<-- [end:polynomial]
print(coefs, line)


# --8<-- [start:powers]
x = np.linspace(1, 10, 10)
y = 2 / x + 0.5 * x

coefs = processing.quick_fit(x, y, degrees=[-1, 1])  # coefficients of x^-1 and x^1
curve = processing.make_curve(x, coefs, degrees=[-1, 1])
# --8<-- [end:powers]
print(coefs)


# --8<-- [start:linhall]
coefs, fitted = data.etr.fit_linhall('rho_xy', x_range=(11, None))  # fit B >= 11 T
intercept, slope = coefs           # same order as quick_fit
print(fitted.head(3).ms)           # new column rho_xy_linHall over the whole field range
print(etr.hall_fit_to_str(coefs))  # a0 = ... ohm*m, k = ... ohm*m/T
# --8<-- [end:linhall]


# --8<-- [start:multiband]
p0 = [1e26, 0.015, 1e25, 0.02]  # n1, mu1, n2, mu2 (interleaved), in m^-3 and m^2/(V*s)
p, fitted = data.etr.fit_multiband(
    p0, cols=['rho_xx', 'rho_xy'], kinds=['xx', 'xy'],
    bands='he',                                # band 1 holes, band 2 electrons
    field_range=(-6, 6), inside_range=False,   # fit only |B| >= 6 T
    report=True,                               # or a path to write the report to
)
n1, mu1, n2, mu2 = p
print(fitted.head(3).ms)  # new columns rho_xx_mbndhe and rho_xy_mbndhe, over all B
# --8<-- [end:multiband]


# --8<-- [start:constrain]
p0 = [1e26, 0.015, 1e25, 0.02]
p, _ = data.etr.fit_multiband(
    p0, cols=['rho_xx', 'rho_xy'], kinds=['xx', 'xy'], bands='he',
    field_range=(-6, 6), inside_range=False,
    bounds={'mu1': (1e-3, 0.05)},  # in m^2/(V*s); unset ones keep the defaults
    expr={'n2': 'n1'},             # compensated: n2 = n1 (in log10 space)
    sigmas=[2e-8, 1e-8],           # uncertainty per column (scalar or per point), ohm*m
)
p_fixed, _ = data.etr.fit_multiband(
    [2.5e25, 0.015, 1e25, 0.02], cols=['rho_xx', 'rho_xy'], kinds=['xx', 'xy'], bands='he',
    field_range=(-6, 6), inside_range=False,
    fix_params={'n1': True},       # n1 stays at its starting value, 2.5e25
)
# --8<-- [end:constrain]
print(p, p_fixed)


# --8<-- [start:calculate]
p = [2.4e25, 0.011, 5.2e24, 0.019]  # n1, mu1, n2, mu2
model = data.etr.calculate_multiband(p, cols=['rho_xx', 'rho_xy'], kinds=['xx', 'xy'],
                                     bands='he', append='model')
print(model.head(3).ms)  # new columns rho_xx_modelhe and rho_xy_modelhe

field = np.linspace(0, 14, 8)  # on a plain array
rho_xy = etr.generate_multiband_eq('xy', 'he')(field, *p)
# --8<-- [end:calculate]
print(rho_xy)
