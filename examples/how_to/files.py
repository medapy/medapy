# --8<-- [start:setup]
from pathlib import Path

from medapy.collection import (MeasurementCollection, MeasurementFile, DefinitionsLoader,
                               ParameterDefinition, ContactPair)

data_dir = Path(__file__).parent.parent / 'files'
definitions = DefinitionsLoader().get_all()  # magnetic_field, temperature, position
collection = MeasurementCollection(data_dir, definitions, file_pattern='*.csv')
# --8<-- [end:setup]


# --8<-- [start:inspect]
print(len(collection))         # number of files
print(collection.head(3))      # numbered listing of the first 3 filenames
print(collection.tail(3))
for mfile in collection[:3]:   # a slice is a collection too
    print(mfile.name)
# --8<-- [end:inspect]


# --8<-- [start:definitions]
gate = ParameterDefinition(
    name_id='gate_voltage',
    long_names=['gate', 'Gate'],
    short_names=['Vg'],
    units=['V'],                # prefixed units (mV, uV) are accepted too
)
name = 'sample_I1-5(1uA)_V2-3_Vg=500mV_T=2K.csv'

only_gate = MeasurementFile(name, parameters=[gate])
print(only_gate.parameters.keys())             # only gate_voltage is parsed

with_defaults = MeasurementFile(name, parameters=definitions + [gate])
print(with_defaults.value_of('gate_voltage'))  # 0.5 (V)
print(with_defaults.value_of('temperature'))   # 2.0 (K)
# --8<-- [end:definitions]


# --8<-- [start:fixed]
at_10K = collection.filter(temperature=10)
from_2_to_7K = collection.filter(temperature=(2, 7))      # range, bounds included
from_20K = collection.filter(temperature=(20, None))      # open range
in_plane_side = collection.filter(position=(45, 'IP'))    # special values work in ranges
in_plane_too = collection.filter(position=[45, 'IP'])     # a list is a range too: 45..90 deg
at_0_or_90 = collection.filter(position=0) + collection.filter(position=90)
# --8<-- [end:fixed]
print(len(at_10K), len(from_2_to_7K), len(from_20K), len(in_plane_side), len(in_plane_too),
      len(at_0_or_90))


# --8<-- [start:sweeps]
field_sweeps = collection.filter(sweeps='magnetic_field')   # any direction
down = collection.filter(sweeps='magnetic_field', sweep_directions='dec')  # or 'down'
up = collection.filter(sweeps='magnetic_field', sweep_directions='inc')    # or 'up'
full_range = collection.filter(magnetic_field_sweep=(-14, 14))  # exactly -14..14 T, either way
# --8<-- [end:sweeps]
print(len(field_sweeps), len(down), len(up), len(full_range))


# --8<-- [start:contacts]
on_2_3 = collection.filter(contacts=(2, 3))                 # pair 2-3, any polarization
# contacts=4 would match a single-contact entry such as V4 (one potential measured)
on_both = collection.filter(contacts=[(2, 3), (2, 6)])      # every listed pair present
at_1uA = collection.filter(contacts=ContactPair(1, 5, 'I', 1e-6))
at_1uA_too = collection.filter(contacts='I1-5(1uA)')        # same, parsed from a string
current_biased = collection.filter(polarization='current')  # or 'I'; 'voltage' or 'V'
# --8<-- [end:contacts]
print(len(on_2_3), len(on_both), len(at_1uA), len(at_1uA_too), len(current_biased))


# --8<-- [start:name]
hall = collection.filter(name_contains='Rxy')
hall_3K = collection.filter(name_contains=['Rxy', r'T=3\.0K'])  # all patterns must match
# --8<-- [end:name]
print(len(hall), len(hall_3K))


# --8<-- [start:combine]
# filter: a file must match every criterion
xx_low_T_up = collection.filter(name_contains='Rxx', temperature=(2, 10),
                                sweeps='magnetic_field', sweep_directions='inc')

# filter_generator(mode='any'): a file must match at least one criterion
any_match = collection.filter_generator(temperature=3, position=90, mode='any')
at_3K_or_90deg = MeasurementCollection(list(any_match), definitions)
# --8<-- [end:combine]
print(len(xx_low_T_up), len(at_3K_or_90deg))


# --8<-- [start:exclude]
no_test_files = collection.exclude(name_contains='twoband')
# exclude drops a file that matches ANY of the criteria
neither = collection.exclude(name_contains='twoband', position=90)
# filter(..., exclude=True) drops only files that match ALL of them
not_hall_at_90 = collection.filter(name_contains='Rxy', position=90, exclude=True)
# --8<-- [end:exclude]
print(len(no_test_files), len(neither), len(not_hall_at_90))


# --8<-- [start:generator]
from itertools import islice

matches = collection.filter_generator(temperature=(2, 7))  # files are checked while iterating
for mfile in islice(matches, 3):                           # stop after the first 3 matches
    print(mfile.name)
# --8<-- [end:generator]


# --8<-- [start:sort]
sample_files = collection.filter(name_contains='^sample')  # all have fixed T and position
by_temperature = sample_files.sort('temperature', 'position')
print(by_temperature.head(10))
# --8<-- [end:sort]


# --8<-- [start:parameters]
mfile = MeasurementFile(
    'sample_I1-5(0.5uA)_V2-3_V2-6_Rxx_sweepField_B14to-14T_T=10.0K_Pos=0.0Deg.csv',
    definitions)
print(mfile.value_of('temperature'))      # 10.0; None if the parameter is swept
print(mfile.range_of('magnetic_field'))   # (14.0, -14.0), in sweep order; None if fixed

state = mfile.state_of('magnetic_field')
print(state.is_swept, state.sweep_direction, state.min, state.max, state.unit)
print(mfile.contact_pairs)                # [I1-5(500nA), V2-3, V2-6] as ContactPair objects
# --8<-- [end:parameters]


# --8<-- [start:rename]
mfile = collection.filter(temperature=10)[0].copy()  # copy: the collection shares its files
mfile.rename(prefix='S1', postfix='sym', ext='dat')
print(mfile.path)       # .../S1_<old stem>_sym.dat; nothing on disk changes

print(mfile.generate_filename(prefix='S1'))  # a name built from the parsed parameters
# --8<-- [end:rename]


# --8<-- [start:copy]
import shutil

out_dir = data_dir / 'results' / 'how_to' / 'T10K'
out_dir.mkdir(parents=True, exist_ok=True)

for mfile in collection.filter(temperature=10):
    target = mfile.copy()
    target.rename(directory=out_dir)       # medapy computes the new path...
    shutil.copy2(mfile.path, target.path)  # ...and shutil copies the file
# --8<-- [end:copy]
print(sorted(p.name for p in out_dir.iterdir()))
