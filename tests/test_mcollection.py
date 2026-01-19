"""Tests for MeasurementCollection class."""

import pytest
from pathlib import Path
from decimal import Decimal
from medapy.collection import (
    MeasurementCollection,
    MeasurementFile,
    ParameterDefinition,
    DefinitionsLoader
)


class TestMeasurementCollectionCreation:
    """Test MeasurementCollection creation and initialization."""

    @pytest.fixture
    def basic_definitions(self):
        """Create basic parameter definitions for testing."""
        return [
            ParameterDefinition(
                name_id="temperature",
                long_names=["temperature"],
                short_names=["T"],
                units=["K"]
            ),
            ParameterDefinition(
                name_id="magnetic_field",
                long_names=["field"],
                short_names=["B"],
                units=["T"]
            ),
        ]

    @pytest.fixture
    def sample_files(self, basic_definitions):
        """Create sample MeasurementFile objects."""
        return [
            MeasurementFile("file1_T=4.2K_B=0T.csv", basic_definitions),
            MeasurementFile("file2_T=300K_B=1T.csv", basic_definitions),
            MeasurementFile("file3_T=77K_sweepField.csv", basic_definitions),
        ]

    def test_create_from_iterable(self, sample_files, basic_definitions):
        """Test creating collection from iterable of MeasurementFile objects."""
        collection = MeasurementCollection(
            collection=sample_files,
            parameters=basic_definitions
        )
        assert len(collection) == 3
        assert collection.files == sample_files

    def test_create_empty_collection(self, basic_definitions):
        """Test creating empty collection from empty iterable."""
        collection = MeasurementCollection(
            collection=[],
            parameters=basic_definitions
        )
        assert len(collection) == 0
        assert collection.files == []

    def test_create_with_custom_separator(self, sample_files, basic_definitions):
        """Test creating collection with custom separator."""
        collection = MeasurementCollection(
            collection=sample_files,
            parameters=basic_definitions,
            separator="-"
        )
        assert collection.separator == "-"

    def test_parameter_definitions_stored(self, sample_files, basic_definitions):
        """Test that parameter definitions are stored correctly."""
        collection = MeasurementCollection(
            collection=sample_files,
            parameters=basic_definitions
        )
        assert "temperature" in collection.param_definitions
        assert "magnetic_field" in collection.param_definitions
        assert len(collection.param_definitions) == 2

    def test_create_from_invalid_type_raises_error(self, basic_definitions):
        """Test that creating collection from invalid type raises ValueError."""
        with pytest.raises(ValueError, match="collection can be str, Path, or Iterable"):
            MeasurementCollection(
                collection=123,  # Invalid type
                parameters=basic_definitions
            )

    def test_create_with_non_mfile_iterable_raises_error(self, basic_definitions):
        """Test that creating collection with non-MeasurementFile items raises error."""
        with pytest.raises(TypeError, match="All items in collection must be"):
            MeasurementCollection(
                collection=["not", "measurement", "files"],
                parameters=basic_definitions
            )


class TestMeasurementCollectionMagicMethods:
    """Test magic methods for MeasurementCollection."""

    @pytest.fixture
    def collection(self, sample_files, basic_definitions):
        """Create a collection for testing."""
        return MeasurementCollection(
            collection=sample_files,
            parameters=basic_definitions
        )

    @pytest.fixture
    def sample_files(self, basic_definitions):
        """Create sample MeasurementFile objects."""
        return [
            MeasurementFile("file1_T=4.2K_B=0T.csv", basic_definitions),
            MeasurementFile("file2_T=300K_B=1T.csv", basic_definitions),
            MeasurementFile("file3_T=77K_sweepField.csv", basic_definitions),
        ]

    @pytest.fixture
    def basic_definitions(self):
        """Basic parameter definitions."""
        return [
            ParameterDefinition(
                name_id="temperature",
                long_names=["temperature"],
                short_names=["T"],
                units=["K"]
            ),
            ParameterDefinition(
                name_id="magnetic_field",
                long_names=["field"],
                short_names=["B"],
                units=["T"]
            ),
        ]

    def test_iter(self, collection):
        """Test iteration over collection."""
        files = list(collection)
        assert len(files) == 3
        assert all(isinstance(f, MeasurementFile) for f in files)

    def test_len(self, collection):
        """Test length of collection."""
        assert len(collection) == 3

    def test_getitem_index(self, collection, sample_files):
        """Test indexing by integer."""
        assert collection[0] == sample_files[0]
        assert collection[1] == sample_files[1]
        assert collection[-1] == sample_files[-1]

    def test_getitem_slice(self, collection, basic_definitions):
        """Test slicing returns new MeasurementCollection."""
        sliced = collection[0:2]
        assert isinstance(sliced, MeasurementCollection)
        assert len(sliced) == 2
        assert sliced[0].name == "file1_T=4.2K_B=0T.csv"
        assert sliced[1].name == "file2_T=300K_B=1T.csv"

    def test_setitem(self, collection, basic_definitions):
        """Test item assignment."""
        new_file = MeasurementFile("new_file_T=10K.csv", basic_definitions)
        collection[0] = new_file
        assert collection[0] == new_file

    def test_setitem_wrong_type_raises_error(self, collection):
        """Test that setting non-MeasurementFile raises TypeError."""
        with pytest.raises(TypeError, match="Can only assign MeasurementFile objects"):
            collection[0] = "not a file"

    def test_delitem(self, collection):
        """Test item deletion."""
        initial_len = len(collection)
        del collection[0]
        assert len(collection) == initial_len - 1

    def test_contains(self, collection, sample_files, basic_definitions):
        """Test membership testing with 'in' operator."""
        assert sample_files[0] in collection

        new_file = MeasurementFile("not_in_collection.csv", basic_definitions)
        assert new_file not in collection

    def test_copy(self, collection, basic_definitions):
        """Test copying collection."""
        copy = collection.copy()
        assert isinstance(copy, MeasurementCollection)
        assert len(copy) == len(collection)
        assert copy is not collection
        assert copy.files is not collection.files
        assert copy == collection

    def test_add_collections(self, collection, basic_definitions):
        """Test adding two collections."""
        other_files = [
            MeasurementFile("other1_T=10K.csv", basic_definitions),
            MeasurementFile("other2_T=20K.csv", basic_definitions),
        ]
        other = MeasurementCollection(other_files, basic_definitions)

        combined = collection + other
        assert isinstance(combined, MeasurementCollection)
        assert len(combined) == len(collection) + len(other)

    def test_add_with_non_collection_raises_error(self, collection):
        """Test that adding non-MeasurementCollection raises TypeError."""
        with pytest.raises(TypeError, match="Cannot add .* to MeasurementCollection"):
            result = collection + [1, 2, 3]

    def test_eq_same_collections(self, collection, sample_files, basic_definitions):
        """Test equality of identical collections."""
        other = MeasurementCollection(sample_files, basic_definitions)
        assert collection == other

    def test_eq_different_collections(self, collection, basic_definitions):
        """Test inequality of different collections."""
        other_files = [MeasurementFile("different.csv", basic_definitions)]
        other = MeasurementCollection(other_files, basic_definitions)
        assert collection != other

    def test_str_short_collection(self, collection):
        """Test string representation of short collection."""
        str_repr = str(collection)
        assert "Filename" in str_repr
        assert "file1_T=4.2K_B=0T.csv" in str_repr
        assert "file2_T=300K_B=1T.csv" in str_repr


class TestMeasurementCollectionFiltering:
    """Test filtering functionality of MeasurementCollection."""

    @pytest.fixture
    def loader(self):
        """Parameter definitions loader."""
        return DefinitionsLoader()

    @pytest.fixture
    def collection(self, loader):
        """Create collection with diverse files for filtering tests."""
        files = [
            MeasurementFile("sample_I1-2_T=4.2K_B=0T_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_I1-2_T=300K_B=0T_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_V3-4_T=4.2K_sweepField_Rxy.csv", loader.get_all()),
            MeasurementFile("sample_I1-2_T=77K_B-14to14T_Rxx.csv", loader.get_all()),
            MeasurementFile("device_V5-6_T=4.2K_B=1T_Rxx.csv", loader.get_all()),
        ]
        return MeasurementCollection(files, loader.get_all())

    def test_filter_by_contacts(self, collection):
        """Test filtering by contact pairs."""
        filtered = collection.filter(contacts=(1, 2))
        assert len(filtered) == 3
        for f in filtered:
            assert f.check_contacts((1, 2))

    def test_filter_by_polarization(self, collection):
        """Test filtering by polarization."""
        filtered = collection.filter(polarization='I')
        assert len(filtered) == 3
        for f in filtered:
            assert f.check_polarization('I')

    def test_filter_by_parameter_exact(self, collection):
        """Test filtering by exact parameter value."""
        filtered = collection.filter(temperature=4.2)
        assert len(filtered) == 3
        for f in filtered:
            assert f.check_parameter("temperature", 4.2)

    def test_filter_by_parameter_range(self, collection):
        """Test filtering by parameter range."""
        filtered = collection.filter(temperature=(4.0, 80.0))
        assert len(filtered) == 4  # Three 4.2K files and one 77K file

    def test_filter_by_sweep(self, collection):
        """Test filtering by swept parameter."""
        filtered = collection.filter(sweeps="magnetic_field")
        assert len(filtered) == 2  # One sweepField, one B-14to14T

    def test_filter_by_name_contains(self, collection):
        """Test filtering by filename substring."""
        filtered = collection.filter(name_contains="device")
        assert len(filtered) == 1
        assert "device" in filtered[0].name

    def test_filter_multiple_conditions(self, collection):
        """Test filtering with multiple conditions (AND logic)."""
        filtered = collection.filter(
            contacts=(1, 2),
            temperature=(4.0, 80.0)
        )
        assert len(filtered) == 2  # I1-2 files with T=4.2K and T=77K

    def test_filter_generator(self, collection):
        """Test filter_generator returns generator."""
        gen = collection.filter_generator(temperature=4.2)
        assert hasattr(gen, '__iter__')
        assert hasattr(gen, '__next__')
        files = list(gen)
        assert len(files) == 3

    def test_exclude_by_contacts(self, collection):
        """Test excluding files by contacts."""
        excluded = collection.exclude(contacts=(1, 2))
        assert len(excluded) == 2  # Should exclude 3 I1-2 files, leaving 2
        for f in excluded:
            assert not f.check_contacts((1, 2))

    def test_exclude_by_polarization(self, collection):
        """Test excluding files by polarization."""
        excluded = collection.exclude(polarization='I')
        assert len(excluded) == 2  # Exclude 3 current-polarized files
        for f in excluded:
            assert not f.check_polarization('I')

    def test_exclude_by_name(self, collection):
        """Test excluding files by name pattern."""
        excluded = collection.exclude(name_contains="device")
        assert len(excluded) == 4  # Exclude 1 file with "device"
        for f in excluded:
            assert "device" not in f.name

    def test_filter_returns_new_collection(self, collection):
        """Test that filter returns new collection without modifying original."""
        original_len = len(collection)
        filtered = collection.filter(temperature=4.2)
        assert len(collection) == original_len
        assert filtered is not collection


class TestMeasurementCollectionManipulation:
    """Test manipulation methods of MeasurementCollection."""

    @pytest.fixture
    def basic_definitions(self):
        """Basic parameter definitions."""
        return [
            ParameterDefinition(
                name_id="temperature",
                long_names=["temperature"],
                short_names=["T"],
                units=["K"]
            ),
            ParameterDefinition(
                name_id="magnetic_field",
                long_names=["field"],
                short_names=["B"],
                units=["T"]
            ),
        ]

    @pytest.fixture
    def collection(self, basic_definitions):
        """Create collection for manipulation tests."""
        files = [
            MeasurementFile("file_T=300K_B=0T.csv", basic_definitions),
            MeasurementFile("file_T=77K_B=1T.csv", basic_definitions),
            MeasurementFile("file_T=4.2K_B=2T.csv", basic_definitions),
        ]
        return MeasurementCollection(files, basic_definitions)

    def test_append(self, collection, basic_definitions):
        """Test appending a file to collection."""
        new_file = MeasurementFile("new_file_T=10K.csv", basic_definitions)
        initial_len = len(collection)
        collection.append(new_file)
        assert len(collection) == initial_len + 1
        assert collection[-1] == new_file

    def test_append_wrong_type_raises_error(self, collection):
        """Test that appending non-MeasurementFile raises TypeError."""
        with pytest.raises(TypeError, match="Can only append MeasurementFile objects"):
            collection.append("not a file")

    def test_extend(self, collection, basic_definitions):
        """Test extending collection from iterable."""
        new_files = [
            MeasurementFile("new1_T=10K.csv", basic_definitions),
            MeasurementFile("new2_T=20K.csv", basic_definitions),
        ]
        initial_len = len(collection)
        collection.extend(new_files)
        assert len(collection) == initial_len + 2

    def test_extend_wrong_type_raises_error(self, collection):
        """Test that extending with non-MeasurementFile items raises error."""
        with pytest.raises(TypeError, match="All items in iterable must be"):
            collection.extend(["not", "files"])

    def test_pop_default(self, collection):
        """Test popping last item."""
        initial_len = len(collection)
        last_file = collection.files[-1]
        popped = collection.pop()
        assert len(collection) == initial_len - 1
        assert popped == last_file

    def test_pop_index(self, collection):
        """Test popping item at specific index."""
        first_file = collection.files[0]
        popped = collection.pop(0)
        assert popped == first_file
        assert len(collection) == 2

    def test_pop_empty_collection_raises_error(self, basic_definitions):
        """Test that popping from empty collection raises IndexError."""
        empty = MeasurementCollection([], basic_definitions)
        with pytest.raises(IndexError):
            empty.pop()

    def test_sort_by_single_parameter(self, collection):
        """Test sorting by single parameter."""
        sorted_coll = collection.sort("temperature")
        temps = [f.parameters["temperature"].state.value for f in sorted_coll]
        assert temps == [Decimal('4.2'), Decimal('77'), Decimal('300')]

    def test_sort_by_multiple_parameters(self, collection):
        """Test sorting by multiple parameters."""
        sorted_coll = collection.sort("magnetic_field", "temperature")
        # Sort by B first (0, 1, 2), then by T
        fields = [f.parameters["magnetic_field"].state.value for f in sorted_coll]
        assert fields == [Decimal('0'), Decimal('1'), Decimal('2')]

    def test_sort_returns_new_collection(self, collection):
        """Test that sort returns new collection without modifying original."""
        original_first = collection[0]
        sorted_coll = collection.sort("temperature")
        assert collection[0] == original_first  # Original unchanged
        assert sorted_coll is not collection

    def test_to_list(self, collection):
        """Test converting collection to list."""
        files_list = collection.to_list()
        assert isinstance(files_list, list)
        assert len(files_list) == len(collection)
        assert all(isinstance(f, MeasurementFile) for f in files_list)
        # Verify it's a copy
        assert files_list is not collection.files


class TestMeasurementCollectionDisplay:
    """Test display methods of MeasurementCollection."""

    @pytest.fixture
    def basic_definitions(self):
        """Basic parameter definitions."""
        return [
            ParameterDefinition(
                name_id="temperature",
                long_names=["temperature"],
                short_names=["T"],
                units=["K"]
            ),
        ]

    @pytest.fixture
    def small_collection(self, basic_definitions):
        """Create small collection for display tests."""
        files = [
            MeasurementFile(f"file{i}_T={i}K.csv", basic_definitions)
            for i in range(5)
        ]
        return MeasurementCollection(files, basic_definitions)

    @pytest.fixture
    def large_collection(self, basic_definitions):
        """Create large collection (>60 files) for display tests."""
        files = [
            MeasurementFile(f"file{i}_T={i}K.csv", basic_definitions)
            for i in range(70)
        ]
        return MeasurementCollection(files, basic_definitions)

    def test_head(self, small_collection, capsys):
        """Test head method displays first n files."""
        small_collection.head(3)
        captured = capsys.readouterr()
        assert "file0_T=0K.csv" in captured.out
        assert "file1_T=1K.csv" in captured.out
        assert "file2_T=2K.csv" in captured.out
        assert "file3_T=3K.csv" not in captured.out

    def test_tail(self, small_collection, capsys):
        """Test tail method displays last n files."""
        small_collection.tail(2)
        captured = capsys.readouterr()
        assert "file3_T=3K.csv" in captured.out
        assert "file4_T=4K.csv" in captured.out
        assert "file0_T=0K.csv" not in captured.out

    def test_str_small_collection(self, small_collection):
        """Test string representation of small collection."""
        str_repr = str(small_collection)
        # Should show all files
        assert "file0_T=0K.csv" in str_repr
        assert "file4_T=4K.csv" in str_repr
        assert "..." not in str_repr

    def test_str_large_collection(self, large_collection):
        """Test string representation of large collection."""
        str_repr = str(large_collection)
        # Should show first 5 and last 5 with ellipsis
        assert "file0_T=0K.csv" in str_repr
        assert "file4_T=4K.csv" in str_repr
        assert "..." in str_repr
        assert "file69_T=69K.csv" in str_repr


class TestMeasurementCollectionIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.fixture
    def loader(self):
        """Parameter definitions loader."""
        return DefinitionsLoader()

    @pytest.fixture
    def realistic_collection(self, loader):
        """Create realistic collection for integration tests."""
        files = [
            MeasurementFile("sample_I1-2(10mA)_V3-4_T=4.2K_B=0T_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_I1-2(10mA)_V3-4_T=4.2K_sweepField_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_I1-2(10mA)_V3-4_T=77K_B=0T_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_I1-2(10mA)_V3-4_T=300K_B=0T_Rxx.csv", loader.get_all()),
            MeasurementFile("sample_V5-6_T=4.2K_B-14to14T_Rxy.csv", loader.get_all()),
        ]
        return MeasurementCollection(files, loader.get_all())

    def test_filter_sort_workflow(self, realistic_collection):
        """Test filtering then sorting workflow."""
        # Filter for low temperature files
        filtered = realistic_collection.filter(temperature=(None, 100))
        assert len(filtered) == 4  # Three 4.2K files and one 77K file

        # Sort by temperature
        sorted_filtered = filtered.sort("temperature")
        temps = [f.parameters["temperature"].state.value for f in sorted_filtered]
        assert temps[0] == Decimal('4.2')
        assert temps[-1] == Decimal('77')

    def test_complex_filtering(self, realistic_collection):
        """Test complex filtering with multiple criteria."""
        # Filter for specific contacts, temperature, and measurement type
        filtered = realistic_collection.filter(
            contacts=(1, 2),
            temperature=4.2,
            name_contains="Rxx"
        )
        assert len(filtered) == 2  # Two matching files

    def test_copy_and_modify(self, realistic_collection, loader):
        """Test copying collection and modifying the copy."""
        copy = realistic_collection.copy()
        original_len = len(realistic_collection)

        # Modify copy
        new_file = MeasurementFile("new_file_T=10K.csv", loader.get_all())
        copy.append(new_file)

        # Original should be unchanged
        assert len(realistic_collection) == original_len
        assert len(copy) == original_len + 1

    def test_combine_collections(self, realistic_collection, loader):
        """Test combining multiple collections."""
        # Create second collection
        other_files = [
            MeasurementFile("device_A_T=4.2K_B=0T.csv", loader.get_all()),
            MeasurementFile("device_B_T=4.2K_B=0T.csv", loader.get_all()),
        ]
        other_collection = MeasurementCollection(other_files, loader.get_all())

        # Combine
        combined = realistic_collection + other_collection
        assert len(combined) == len(realistic_collection) + len(other_collection)

    def test_filter_exclude_combination(self, realistic_collection):
        """Test using filter and exclude together."""
        # Get low temperature files
        low_temp = realistic_collection.filter(temperature=(None, 100))

        # Exclude swept measurements
        fixed_field = low_temp.exclude(sweeps="magnetic_field")

        # Should have 4.2K and 77K files with fixed B field
        assert len(fixed_field) == 2

    def test_slice_and_iterate(self, realistic_collection):
        """Test slicing and iteration."""
        # Get first 3 files
        subset = realistic_collection[0:3]
        assert len(subset) == 3

        # Iterate over subset
        count = 0
        for f in subset:
            assert isinstance(f, MeasurementFile)
            count += 1
        assert count == 3

    def test_modify_collection_in_place(self, realistic_collection, loader):
        """Test in-place modification of collection."""
        original_len = len(realistic_collection)

        # Append
        new_file = MeasurementFile("new_T=10K.csv", loader.get_all())
        realistic_collection.append(new_file)
        assert len(realistic_collection) == original_len + 1

        # Pop
        popped = realistic_collection.pop()
        assert popped == new_file
        assert len(realistic_collection) == original_len

        # Delete
        del realistic_collection[0]
        assert len(realistic_collection) == original_len - 1
