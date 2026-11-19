# Processing

## `.proc` accessor

Accessed as `df.proc` on any `DataFrame` after `from medapy.analysis import proc_pandas`.
The sheet must be initialized with `df.ms.init_msheet()`; methods act on the `.ms` axes,
with `df.proc.x` / `df.proc.col_x` (and `y`, `z`) as shortcuts to the axis data and column
names. The same methods are available on [`df.etr`](transport.md).

::: medapy.analysis.proc_pandas.DataProcessingAccessor
    options:
      heading_level: 3

## Array functions

The engine behind `.proc`, usable on plain arrays and DataFrames:
`from medapy.analysis import processing`. `savgol_filter` needs the `fit` extra (`scipy`).

::: medapy.analysis.processing
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
