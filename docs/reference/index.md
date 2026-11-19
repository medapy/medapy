# API Reference

Generated from the docstrings. The pages are grouped by what you work with:

| Page | Covers | Reached through |
| --- | --- | --- |
| [Measurement sheet](ms.md) | The `.ms` accessor: units, labels, axes, saving | `from medapy import ms_pandas`, then `df.ms` |
| [Processing](processing.md) | The `.proc` accessor and the array functions behind it | `from medapy.analysis import proc_pandas`, then `df.proc` |
| [Electron transport](transport.md) | The `.etr` accessor and the transport functions | `from medapy.analysis import etr`, then `df.etr` or `etr.<function>` |
| [Collections](collection.md) | Measurement files, collections, contact pairs, parameters | `from medapy import MeasurementCollection, ...` |
| [Utilities](utilities.md) | SI prefixes and warning classes | `medapy.utils.prefixes`, `medapy.utils.warnings` |

The accessor classes are never instantiated directly: importing the module registers the
accessor with pandas, and pandas creates it on attribute access. See
[The accessor family](../explanation.md#the-accessor-family) for how the three relate.

The shared unit registry is `medapy.ureg`; always import it from the package root
(`from medapy import ureg`). See [Units and the registry](../explanation.md#units-and-the-registry).
