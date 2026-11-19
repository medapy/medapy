# MeDaPy

**Me**asurement **Da**ta **Py**thon package: unit-aware pandas tools for measured data, with a focus on electron transport.

- **Measurement sheets** (`df.ms`): units parsed from column names, labels and axes for convenient access, unit conversion and unit-aware arithmetic via [Pint](https://pint.readthedocs.io).
- **Collections**: build a set of measurement files from a folder and filter it by parameters parsed from the filenames (temperature, field range, contacts, ...).
- **Processing** (`df.proc`): range selection, symmetrization, normalization, interpolation, smoothing.
- **Electron transport** (`df.etr`): resistance to resistivity, linear Hall and two-band/multiband fits.

> [!NOTE]
> MeDaPy is in alpha; the API may still change.

**Documentation:** https://medapy.github.io/medapy/

## Installation

```bash
pip install medapy          # core: numpy, pandas, pint, pint-pandas
pip install "medapy[fit]"   # + scipy, lmfit, numdifftools (needed for df.etr and fitting)
pip install "medapy[plot]"  # + matplotlib (publication style and colormaps)
pip install "medapy[all]"   # fit + plot
```

MeDaPy requires Python 3.11 or newer.

## Quickstart

```python
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
```

Importing `ms_pandas` and `proc_pandas` registers the `.ms` and `.proc` accessors on every DataFrame. The printed sheet shows each column's unit, axis and labels above the data.

## Learn more

- [Tutorials](https://medapy.github.io/medapy/tutorials/): a guided path from measurement-sheet basics to a complete fit-and-save workflow.
- [API reference](https://medapy.github.io/medapy/reference/)
- [`examples/`](https://github.com/medapy/medapy/tree/master/examples): the runnable scripts the tutorials are built from, with sample data.

## Development

MeDaPy uses [uv](https://docs.astral.sh/uv/). From a clone:

```bash
uv sync --all-extras    # editable install with all extras and the dev tools
uv run pytest
```

## License

MIT, see [LICENSE](https://github.com/medapy/medapy/blob/master/LICENSE).
