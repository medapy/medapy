# Electron transport

Requires the `fit` extra (`pip install "medapy[fit]"`).

## `.etr` accessor

Accessed as `df.etr` on any `DataFrame` after `from medapy.analysis import etr`.
It subclasses [`DataProcessingAccessor`](processing.md#medapy.analysis.proc_pandas.DataProcessingAccessor),
so every `.proc` method is also available on `.etr`; only the transport methods are listed here.

Note the parameter order: `fit_twoband` / `calculate_twoband` take `[n1, n2, mu1, mu2]`, while
`fit_multiband` / `calculate_multiband` take them interleaved as `[n1, mu1, n2, mu2, ...]`.

::: medapy.analysis.electron_transport.etr_pandas.ElectricalTransportAccessor
    options:
      heading_level: 3

[](){#medapy.analysis.electron_transport.transport}
## Transport functions

The same `from medapy.analysis import etr` import returns this module, so the functions are
called as `etr.fit_twoband(...)`, `etr.r2rho(...)` and so on, on plain arrays.

::: medapy.analysis.electron_transport.transport
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
