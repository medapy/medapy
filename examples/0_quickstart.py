# --8<-- [start:quickstart]
import numpy as np
import pandas as pd
from medapy import ms_pandas
from medapy.analysis import proc_pandas

field = np.linspace(-9, 9, 19)
df = pd.DataFrame({'Field (T)': field,
                   'Resistance (ohm)': 100 + 0.5 * field**2 + 2 * field})

df.ms.init_msheet()  # parse units from column names; x = Field, y = Resistance
df.ms.add_labels({'Field': 'B', 'Resistance': 'R'})
df.ms.convert_unit('B', 'mT')  # converts values and unit together
df.proc.symmetrize(inplace=True)  # keep the part of R even in B
print(df.ms)
# --8<-- [end:quickstart]
