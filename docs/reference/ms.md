# Measurement sheet

Accessed as `df.ms` on any `DataFrame` after `from medapy import ms_pandas`.

Besides the methods below, `df.ms['label']` returns a column by label or name, and the axes
are attributes: `df.ms.x` is the data of the x-axis column (likewise `y`, `z` and custom axes).

::: medapy.ms_pandas.MeasurementSheetAccessor
    options:
      filters: ["!^_", "^__(get|set)item__$"]
