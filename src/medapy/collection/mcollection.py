from typing import Iterator, Union, Tuple, Iterable
from pathlib import Path

from medapy.collection import (MeasurementFile,
                               ParameterDefinition)
from medapy.utils import validations

class MeasurementCollection:
    """
    Manage and filter collections of measurement files.

    Provides methods to filter files by measurement parameters, contact configurations,
    and other criteria. Supports iteration, slicing, and collection operations.

    Parameters
    ----------
    collection : str, Path, or iterable of MeasurementFile
        Source of measurement files:
        - str or Path: directory path to scan for files
        - iterable: existing collection of MeasurementFile objects
    parameters : iterable of ParameterDefinition
        Parameter definitions used to parse filenames.
    file_pattern : str, default "*.*"
        Glob pattern for matching files in directory (used only if collection is a path).
    separator : str, default "_"
        Separator character used in filenames.

    Attributes
    ----------
    files : list of MeasurementFile
        List of measurement files in the collection.
    param_definitions : dict
        Parameter definitions keyed by name_id.
    separator : str
        Filename separator character.
    folder_path : Path, optional
        Path to the folder if collection was created from directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> from medapy.collection import MeasurementCollection, DefinitionsLoader
    >>> # Load default parameter definitions
    >>> parameters = DefinitionsLoader().get_all()
    >>> # Create collection from directory
    >>> path = Path('examples/files')
    >>> collection = MeasurementCollection(path, parameters)
    >>> # Explore collection
    >>> collection.head(5)  # Show first 5 files
    >>> len(collection)  # Number of files
    >>> # Filter by criteria
    >>> low_temp_files = collection.filter(temperature=(2, 5))
    >>> current_sweep = collection.filter(sweeps='magnetic_field',
    ...                                    polarization='I')
    >>> # Iterate over files
    >>> for mfile in collection:
    ...     print(mfile.name)

    See Also
    --------
    MeasurementFile : Individual measurement file representation
    filter : Filter collection by criteria
    filter_generator : Generator version of filter
    ParameterDefinition : Define parameter schemas
    """
    def __init__(self,
                 collection: str | Path | Iterable[MeasurementFile],
                 parameters: Iterable[ParameterDefinition],
                 file_pattern: str = "*.*",
                 separator: str = "_"):
        """
        Initialize measurement collection from directory or file list.

        Parameters
        ----------
        collection : str, Path, or iterable of MeasurementFile
            Source of measurement files:
            - Directory path (str or Path): scans for files matching file_pattern
            - Iterable of MeasurementFile: uses provided file objects
        parameters : iterable of ParameterDefinition
            Parameter definitions for parsing filenames. All items must be
            ParameterDefinition instances.
        file_pattern : str, default "*.*"
            Glob pattern for file matching when collection is a directory path.
            Examples: "*.csv", "*.dat", "*_T=*K.txt"
        separator : str, default "_"
            Character used to separate filename parts for parsing.

        Examples
        --------
        >>> from pathlib import Path
        >>> from medapy.collection import MeasurementCollection, DefinitionsLoader
        >>> # From directory
        >>> params = DefinitionsLoader().get_all()
        >>> collection = MeasurementCollection(
        ...     Path('data/measurements'),
        ...     parameters=params,
        ...     file_pattern="*.csv"
        ... )
        >>> # From existing file list
        >>> files = [mfile1, mfile2, mfile3]
        >>> collection = MeasurementCollection(files, parameters=params)
        """

        validations.class_in_iterable(parameters, ParameterDefinition, iter_name='parameters')
        self.param_definitions = {param.name_id: param for param in parameters}
        self.separator = separator

        if isinstance(collection, (str, Path)):
            self.folder_path = Path(collection)
            self.files = []
            for f in self.folder_path.glob(file_pattern):
                if f.is_file():
                    self.files.append(MeasurementFile(f, parameters, separator))
            return
        if isinstance(collection, Iterable):
            if not collection:
                self.files = []
                return
            validations.class_in_iterable(collection, MeasurementFile, iter_name='collection')
            self.files = list(collection)
            return
        raise ValueError("collection can be str, Path, or Iterable; "
                         f"got {type(collection)}")

    def __iter__(self) -> Iterator[MeasurementFile]:
        """Iterate over all measurement files"""
        return iter(self.files)

    def __copy__(self):
        """Create a shallow copy of the object"""
        # Create new instance of the same class
        return self.__class__(collection=self.files.copy(),
                              parameters=list(self.param_definitions.values()))

    def __add__(self, other):
        """Enable addition with another Collection or list"""
        if isinstance(other, MeasurementCollection):
            files = self.files + other.files
            parameters = (other.param_definitions | self.param_definitions).values()
            return MeasurementCollection(collection=files,
                                         parameters=parameters)
        raise TypeError(f"Cannot add {type(other)} to MeasurementCollection")

    def __len__(self):
        """Return the number of files in collection"""
        return len(self.files)

    def __getitem__(self, index):
        """Enable indexing and slicing"""
        parameters = self.param_definitions.values()
        if isinstance(index, slice):
            return MeasurementCollection(collection=list(self.files[index]),
                                        parameters=parameters)
        return self.files[index]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.files == other.files and
                    self.param_definitions == other.param_definitions)
        return super().__eq__(other)

    def __setitem__(self, index, value):
        """Enable item assignment"""
        if not isinstance(value, MeasurementFile):
            raise TypeError("Can only assign MeasurementFile objects")
        self.files[index] = value

    def __delitem__(self, index):
        """Enable item deletion"""
        del self.files[index]

    def __contains__(self, item):
        """Enable 'in' operator"""
        return item in self.files

    def __str__(self):
        """String representation"""
        res = self.__get_repr_header()
        length = len(self)
        if length <= 60:
            res += self._head_files_str(length)
        else:
            res += self._head_files_str(5)
            res += f"\n{'..':2}    {'...':^8}\n"
            res += self._tail_files_str(5)
        return res

    # def __repr__(self):
    #     """Detailed string representation"""
    #     res = self.__get_repr_header()
    #     length = len(self)
    #     if length <= 60:
    #         res += self._head_files_str(length)
    #     else:
    #         res = self._head_files_str(5)
    #         res += f'{'..':2}    {'...':^8}\n'
    #         res += self._tail_files_str(5)
    #     return res

    def filter_generator(self,
               contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
               polarization: str | None = None,
               sweeps: list[str] | str | None = None,
               sweep_directions: list[str | None] | str | None = None,
               exact_sweeps: bool = True,
               name_contains: list[str] | str | None = None,
               exclude: bool = False,
               mode: str = 'all',
               **parameter_filters) -> Iterator[MeasurementFile]:
        """
        Filter measurement files based on criteria, yielding results lazily.

        Supports both AND logic (all criteria must match) and OR logic (any criterion
        must match) via the mode parameter. Memory-efficient for large collections.

        Parameters
        ----------
        contacts : various types, optional
            Contact specification (int, tuple, str, ContactPair, or list).
        polarization : {'I', 'V'}, optional
            Filter by polarization type.
        sweeps : str or list of str, optional
            Parameter name(s) that should be swept.
        sweep_directions : {'inc', 'dec'} or list, optional
            Expected sweep direction(s).
        exact_sweeps : bool, default True
            If True, sweep ranges must match exactly.
        name_contains : str or list of str, optional
            String(s) or regex pattern(s) that must appear in filename.
        exclude : bool, default False
            If True, exclude files matching criteria instead of including them.
        mode : {'all', 'any'}, default 'all'
            Logic mode:
            - 'all': AND logic - files must match ALL criteria
            - 'any': OR logic - files must match AT LEAST ONE criterion
        **parameter_filters : keyword arguments
            Filter by parameter values.

        Yields
        ------
        MeasurementFile
            Files matching the filter criteria.

        Examples
        --------
        >>> # AND logic (all criteria must match)
        >>> for mfile in collection.filter_generator(
        ...     polarization='I',
        ...     temperature=(2, 5),
        ...     mode='all'
        ... ):
        ...     print(mfile.name)
        >>> # OR logic (any criterion matches)
        >>> for mfile in collection.filter_generator(
        ...     temperature=4.2,
        ...     magnetic_field=0,
        ...     mode='any'
        ... ):
        ...     print(mfile.name)

        See Also
        --------
        filter : Returns MeasurementCollection (AND logic only)
        exclude : Excludes files matching ANY criterion (OR logic)
        """
        for meas_file in self.files:
            matches = meas_file.check(
                mode=mode,
                contacts=contacts,
                polarization=polarization,
                sweeps=sweeps,
                sweep_directions=sweep_directions,
                exact_sweep=exact_sweeps,
                name_contains=name_contains,
                **parameter_filters
            )
            if matches != exclude:
                yield meas_file

    def filter(self,
               contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
               polarization: str | None = None,
               sweeps: list[str] | str | None = None,
               sweep_directions: list[str | None] | str | None = None,
               exact_sweeps: bool = True,
               name_contains: list[str] | str | None = None,
               exclude: bool = False,
               **parameter_filters) -> 'MeasurementCollection':
        """
        Filter measurement files based on various criteria, returning a new collection
        Uses AND logic - files must match ALL criteria

        Args:
            contacts: Single contact pair (1, 2), list of pairs/contacts [(1, 2), 3], or single contact
            polarization: 'I' for current or 'V' for voltage
            sweep_direction: 'inc' or 'dec'
            exclude: If True, exclude files that match criteria instead of including them
            **parameter_filters: Parameter name with value or (min, max) tuple

        Returns:
            New MeasurementCollection containing only matching files
        """
        filtered_files = list(self.filter_generator(
            contacts=contacts,
            polarization=polarization,
            sweeps=sweeps,
            sweep_directions=sweep_directions,
            exact_sweeps=exact_sweeps,
            name_contains=name_contains,
            exclude=exclude,
            mode='all',
            **parameter_filters
        ))
        return MeasurementCollection(
            filtered_files,
            parameters=self.param_definitions.values(),
            separator=self.separator
        )

    def exclude(self,
                contacts: Union[Tuple[int, int], list[Union[Tuple[int, int], int]], int] = None,
                polarization: str | None = None,
                sweeps: list[str] | str | None = None,
                sweep_directions: list[str | None] | str | None = None,
                exact_sweeps: bool = True,
                name_contains: list[str] | str | None = None,
                **parameter_filters) -> 'MeasurementCollection':
        """
        Exclude files matching ANY of the given criteria (OR logic).

        Unlike filter() which uses AND logic, this method excludes files that match
        any single criterion. Useful for removing unwanted files from collection.

        Parameters
        ----------
        contacts : various types, optional
            Contact specification to exclude.
        polarization : {'I', 'V'}, optional
            Exclude files with this polarization type.
        sweeps : str or list of str, optional
            Exclude files sweeping these parameters.
        sweep_directions : {'inc', 'dec'} or list, optional
            Exclude files with these sweep directions.
        exact_sweeps : bool, default True
            If True, sweep ranges must match exactly.
        name_contains : str or list of str, optional
            Exclude files with these strings/patterns in filename.
        **parameter_filters : keyword arguments
            Exclude files matching these parameter values.

        Returns
        -------
        MeasurementCollection
            New collection excluding files that match ANY criterion (OR logic).

        Examples
        --------
        >>> # Exclude files at zero field OR at 4.2K
        >>> subset = collection.exclude(
        ...     magnetic_field=0,
        ...     temperature=4.2
        ... )
        >>> # File excluded if field=0 OR temp=4.2 (not both required)
        >>> # Exclude voltage polarization files
        >>> current_only = collection.exclude(polarization='V')

        Notes
        -----
        This method uses OR logic: a file is excluded if it matches ANY of the
        specified criteria. To exclude files matching ALL criteria, use:
        ``collection.filter(..., exclude=True)`` instead.

        See Also
        --------
        filter : Include files matching ALL criteria (AND logic)
        filter_generator : Generator with configurable AND/OR logic
        """
        filtered_files = list(self.filter_generator(
            contacts=contacts,
            polarization=polarization,
            sweeps=sweeps,
            sweep_directions=sweep_directions,
            exact_sweeps=exact_sweeps,
            name_contains=name_contains,
            exclude=True,
            mode='any',
            **parameter_filters
        ))
        return MeasurementCollection(
            filtered_files,
            parameters=self.param_definitions.values(),
            separator=self.separator
        )

    def sort(self, *parameters) -> 'MeasurementCollection':
        files = sorted(self.files, key=lambda f: tuple(f.state_of(p).value for p in parameters))
        return type(self)(collection=files.copy(),
                          parameters=list(self.param_definitions.values()))

    def copy(self):
        return self.__copy__()

    def append(self, item) -> None:
        """Add a file to collection"""
        if not isinstance(item, MeasurementFile):
            raise TypeError("Can only append MeasurementFile objects")
        self.files.append(item)

    def extend(self, iterable: Iterable) -> None:
        """Extend collection from iterable"""
        validations.class_in_iterable(iterable, MeasurementFile, iter_name='iterable')
        self.files.extend(iterable)

    def pop(self, index: int = -1):
        """
        Remove and return item at index (default last).
        Raises IndexError if collection is empty or index is out of range.
        """
        return self.files.pop(index)

    def head(self, n: int = 5) -> None:
        header = self.__get_repr_header()
        print(header + self._head_files_str(n))

    def tail(self, n: int = 5) -> None:
        header = self.__get_repr_header()
        print(header + self._tail_files_str(n))

    def to_list(self):
        return self.files.copy()

    def __get_repr_header(self):
        return f"{'':2}    Filename\n"

    def _head_files_str(self, n: int) -> str:
        head = ''
        for (i, f) in enumerate(self.files[:n]):
            head += f'{i:>2}    {f.path.name}\n'
        return head.rstrip('\n')

    def _tail_files_str(self, n: str) -> str:
        tail = ''
        ref_idx = len(self.files) - n
        for (i, f) in enumerate(self.files[-n:]):
            tail += f'{ref_idx + i:>2}    {f.path.name}\n'
        return tail.rstrip('\n')

# Drafts/templates
    # def group_by(self, attribute: str) -> dict[str, list[MeasurementFile]]:
    #     """Group files by given attribute"""
    #     result = {}
    #     for f in self.files:
    #         key = str(getattr(f, attribute))
    #         result.setdefault(key, []).append(f)
    #     return result
