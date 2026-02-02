from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pint

import medapy.utils.misc as misc
from medapy.analysis.proc_pandas import DataProcessingAccessor
from . import electron_transport as etr


ureg = pint.get_application_registry()

@pd.api.extensions.register_dataframe_accessor("etr")
class ElectricalTransportAccessor(DataProcessingAccessor):
    """
    Pandas accessor for electrical transport analysis.

    Extends DataProcessingAccessor with specialized methods for electron transport
    measurements including resistance-to-resistivity conversion, Hall effect analysis,
    and multi-band transport fitting.

    Inherits all data processing methods from DataProcessingAccessor (ensure_increasing,
    select_range, normalize, symmetrize, interpolate, etc.).

    Examples
    --------
    >>> from medapy import ms_pandas, ureg
    >>> from medapy.analysis.electron_transport import etr_pandas
    >>> import pandas as pd
    >>> # Prepare measurement data
    >>> df = pd.DataFrame({
    ...     'Field (T)': [-14, -7, 0, 7, 14],
    ...     'R_xx (Ohm)': [100, 100, 100, 100, 100],
    ...     'R_xy (Ohm)': [-1.4, -0.7, 0, 0.7, 1.4]
    ... })
    >>> df.ms.init_msheet(translations={'Ohm': 'ohm'})
    >>> # Define sample geometry
    >>> t = 50 * ureg('nm')      # Thickness
    >>> w = 10 * ureg('um')      # Width
    >>> l = 2 * ureg('um')       # Length
    >>> # Convert resistance to resistivity
    >>> df.etr.r2rho('xx', col='R_xx', t=t, width=w, length=l,
    ...              new_col='rho_xx', inplace=True)
    >>> df.etr.r2rho('xy', col='R_xy', t=t,
    ...              new_col='rho_xy', inplace=True)
    >>> # Fit linear Hall response
    >>> coefs, _ = df.etr.fit_linhall(col='rho_xy', add_col='fit_lin',
    ...                                inplace=True)
    >>> # Fit two-band model
    >>> p0 = [1e26, 1e25, 0.015, 0.02]  # [n1, n2, mu1, mu2]
    >>> p_opt, _ = df.etr.fit_twoband(p0, col='rho_xy', kind='xy',
    ...                                bands='he', inplace=True)

    See Also
    --------
    DataProcessingAccessor : Base class with general processing methods
    medapy.analysis.electron_transport.electron_transport : Underlying functions
    MeasurementSheetAccessor : Core measurement sheet functionality

    Notes
    -----
    All methods support inplace operations and preserve measurement sheet metadata.
    The `.etr` accessor requires initialization with `.ms.init_msheet()` first.
    """
    def r2rho(self,
              kind: str,
              col: str | None = None,
              *,
              t: float,
              width: float = None,
              length: float = None,
              new_col: str = 'rho',
              set_axis: str | None = None,
              add_label : str | None = None,
              inplace: bool = False
              ) -> pd.DataFrame | None:
        """
        Convert resistance to resistivity using sample geometry.

        Parameters
        ----------
        kind : {'xx', 'xy'}
            Measurement configuration:
            - 'xx': longitudinal (requires width and length)
            - 'xy': Hall/transverse (requires only thickness)
        col : str, optional
            Column name or label containing resistance data.
            If None, uses y-axis column.
        t : float or pint.Quantity
            Sample thickness. If float, assumed to be in meters.
            Can be pint Quantity with units (e.g., `50 * ureg('nm')`).
        width : float or pint.Quantity, optional
            Sample width (required for kind='xx'). If float, assumed meters.
        length : float or pint.Quantity, optional
            Sample length (required for kind='xx'). If float, assumed meters.
        new_col : str, default 'rho'
            Name for the new resistivity column.
        set_axis : str, optional
            Axis assignment for new column (e.g., 'y', 'z').
        add_label : str, optional
            Label to add to the new column.
        inplace : bool, default False
            If True, modify DataFrame in place.

        Returns
        -------
        pd.DataFrame or None
            Modified DataFrame if inplace=False, None otherwise.

        Examples
        --------
        >>> from medapy import ureg
        >>> # Longitudinal resistivity (xx)
        >>> t = 50 * ureg('nm')
        >>> w = 10 * ureg('um')
        >>> l = 2 * ureg('um')
        >>> df.etr.r2rho('xx', col='R_xx', t=t, width=w, length=l,
        ...              new_col='rho_xx', add_label='rho_xx', inplace=True)
        >>> # Hall resistivity (xy) - only needs thickness
        >>> df.etr.r2rho('xy', col='R_xy', t=t,
        ...              new_col='rho_xy', add_label='rho_xy', inplace=True)
        >>> # Using floats (assumed meters)
        >>> df.etr.r2rho('xx', t=50e-9, width=10e-6, length=2e-6,
        ...              new_col='rho', inplace=True)

        Notes
        -----
        - For 'xx': rho = R * (width * thickness) / length
        - For 'xy': rho = R * thickness
        - Units are automatically handled via pint if provided with units
        - Result units are derived from input resistance and geometry units

        See Also
        --------
        fit_linhall : Fit linear Hall response
        fit_twoband : Fit multi-band transport model
        """
        # If t, width, and length are floats, it is assumed they are in meter units

        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)

        if hasattr(t, 'units'):
            t = t.to('m')
            t, t_unit = t.magnitude, t.units
        else:
            t_unit = ureg.Unit('m')

        if hasattr(width, 'units') and hasattr(length, 'units'):
            width = width.to('m').magnitude
            length = length.to('m').magnitude
        elif hasattr(width, 'units') ^ hasattr(length, 'units'): # XOR
            # only one of them has units
            raise AttributeError("Only one of width and length have units")

        # Work on a copy of the data
        df = self._get_df_copy()

        # Should we convert r to 'ohm' before calculating 'rho'?
        r_unit = pint.Unit(df.ms.get_unit(col))
        unit = r_unit * t_unit
        # new_col = f"{new_col}_{kind}"

        # Calculate resistivity values
        new_values = etr.r2rho(df.ms[col], kind=kind, t=t, width=width, length=length)

        # Assign values and metadata
        df.ms._set_column_state(new_col, new_values, unit, set_axis, add_label)

        return self._if_inplace(df, inplace)

    def fit_linhall(self,
                    col: str | None = None,
                    x_range: npt.ArrayLike | None = None,
                    *,
                    add_col: str = 'linHall',
                    set_axis: str | None = None,
                    add_label: str | None = None,
                    inplace: bool = False
                    ) -> tuple[np.ndarray, pd.DataFrame | None]:
        """
        Fit linear Hall response to extract carrier density and mobility.

        Performs linear regression on Hall resistivity vs. magnetic field data.
        Useful for single-carrier systems or high-field limit of multi-carrier systems.

        Parameters
        ----------
        col : str, optional
            Column containing Hall resistivity data. If None, uses y-axis column.
        x_range : array-like of length 2, optional
            Field range (min, max) to use for fitting. If None, uses all data.
        add_col : str, default 'linHall'
            Suffix for new column containing fit values. If empty string,
            no fit column is added.
        set_axis : str, optional
            Axis assignment for fit column.
        add_label : str, optional
            Label for fit column.
        inplace : bool, default False
            If True, modify DataFrame in place.

        Returns
        -------
        coefs : np.ndarray
            Linear fit coefficients [slope, intercept].
        df : pd.DataFrame or None
            Modified DataFrame if inplace=False, None otherwise.

        Examples
        --------
        >>> # Fit Hall data in high-field region
        >>> coefs, _ = df.etr.fit_linhall(
        ...     col='rho_xy',
        ...     x_range=(11, None),  # Field > 11 T
        ...     add_col='fit_lin',
        ...     add_label='flin',
        ...     inplace=True
        ... )
        >>> slope, intercept = coefs
        >>> # Calculate carrier density from slope
        >>> # n = 1 / (e * slope) for 2D systems

        Notes
        -----
        For Hall measurements, the slope is related to carrier density:
        - 2D systems: n_2D = 1 / (e * slope)
        - 3D systems: n_3D = 1 / (e * t * slope) where t is thickness

        See Also
        --------
        fit_twoband : Fit multi-band Hall response
        r2rho : Convert resistance to resistivity
        """
        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)

        # Calculate fit coefficients
        coefs = misc.quick_fit(self.x, self.ms[col], x_range=x_range)

        # Work on a copy of the data
        df = self._get_df_copy()

        if add_col:
            # Prepare metadata
            unit = df.ms.get_unit(col)
            # Prepare new column name
            new_col = self._col_name_append(col, append=add_col)
            # Calculate fit values
            new_values =  misc.make_curve(df.ms.x, coefs)
            # Assign values and metadata
            df.ms._set_column_state(new_col, new_values, unit, set_axis, add_label)

        return coefs, self._if_inplace(df, inplace)

    def fit_twoband(self,
                    p0: tuple[float, float, float, float],
                    col: str | None = None,
                    *,
                    kind: str,
                    bands: str,
                    field_range: npt.ArrayLike | None = None,
                    inside_range: bool = True,
                    extension: tuple | npt.ArrayLike | pd.DataFrame | None = None,
                    add_col: str | None = '2bnd',
                    set_axis: str | None = None, add_label : str | None = None,
                    inplace: bool = False,
                    **kwargs) -> tuple[tuple, pd.DataFrame | None]:
        """
        Fit two-band transport model to extract carrier densities and mobilities.

        Fits magnetoresistance (xx) or Hall (xy) data to a two-band model accounting
        for multiple carrier types (electrons and/or holes) with different densities
        and mobilities.

        Parameters
        ----------
        p0 : tuple of (float, float, float, float)
            Initial guess for fit parameters [n1, n2, mu1, mu2] in SI units:
            - n1, n2: carrier densities (m^-2 for 2D or m^-3 for 3D)
            - mu1, mu2: mobilities (m^2/(V·s))
        col : str, optional
            Column containing resistivity data. If None, uses y-axis column.
        kind : {'xx', 'xy'}
            Measurement type:
            - 'xx': longitudinal magnetoresistance
            - 'xy': Hall resistivity
        bands : str
            Band configuration using letters:
            - 'e': electron band
            - 'h': hole band
            Examples: 'ee' (two electron bands), 'he' (hole + electron),
            'hh' (two hole bands)
        field_range : array-like of length 2, optional
            Magnetic field range (min, max) for fitting. If None, uses all data.
        inside_range : bool, default True
            If True, fit data inside field_range. If False, fit outside.
        extension : tuple, array, or DataFrame, optional
            Additional (field, resistivity) data to include in fit.
            Can be tuple of arrays, DataFrame, or measurement sheet.
        add_col : str, default '2bnd'
            Suffix for new column with fit values. If None, no column added.
        set_axis : str, optional
            Axis assignment for fit column.
        add_label : str, optional
            Label for fit column.
        inplace : bool, default False
            If True, modify DataFrame in place.
        **kwargs
            Additional arguments passed to fitting function (e.g., report=True).

        Returns
        -------
        coefs : tuple
            Optimized parameters (n1, n2, mu1, mu2) in SI units.
        df : pd.DataFrame or None
            Modified DataFrame if inplace=False, None otherwise.

        Examples
        --------
        >>> # Fit Hall resistivity with hole + electron bands
        >>> p0 = [1e26, 1e25, 0.015, 0.02]  # Initial [n1, n2, mu1, mu2]
        >>> p_opt, _ = df.etr.fit_twoband(
        ...     p0,
        ...     col='rho_xy',
        ...     kind='xy',
        ...     bands='he',
        ...     add_col='fit2b',
        ...     report=True,  # Print fit report
        ...     inplace=True
        ... )
        >>> n_hole, n_electron, mu_hole, mu_electron = p_opt
        >>> # Fit magnetoresistance in limited field range
        >>> p_opt_xx, _ = df.etr.fit_twoband(
        ...     p0,
        ...     col='rho_xx',
        ...     kind='xx',
        ...     bands='he',
        ...     field_range=(-6, 6),
        ...     inside_range=False,  # Fit |B| > 6 T
        ...     inplace=True
        ... )

        Notes
        -----
        The two-band model assumes independent carrier types with distinct
        densities and mobilities. Common band configurations:
        - 'he': Compensated semiconductor (hole + electron)
        - 'ee': Two electron bands with different mobilities
        - 'hh': Two hole bands with different mobilities

        See Also
        --------
        fit_linhall : Linear Hall fit (single carrier)
        calculate_twoband : Calculate two-band model with known parameters
        r2rho : Convert resistance to resistivity
        """
        # Default to y axis column if None provided
        col = self.col_y if col is None else self.ms.get_column(col)

        if isinstance(extension, pd.DataFrame):
            if '_ms_axes' in extension.attrs:
                extension = (extension.ms.x, extension.ms.y)
            else:
                extension = (extension.iloc[:, 0], extension.iloc[:, 1])

        # Work with particular columns
        field, rho = self.x, self.ms[col]
        if field_range:
            fldrho = np.column_stack((self.x, self.ms[col]))
            fldrho = misc.select_range_arr(fldrho, 0, field_range, inside_range=inside_range)
            field, rho = fldrho.T

        # Calculate fit coefficient
        coefs = etr.fit_twoband(field, rho, p0, kind=kind, bands=bands, extension=extension, **kwargs)

        # Work on a copy of the data
        df = self._get_df_copy()

        if add_col:
            if kind == 'xx':
                func_2bnd = etr.gen_mr2bnd_eq(bands)
            else:
                func_2bnd = etr.gen_hall2bnd_eq(bands)
            # Prepare metadata
            unit = df.ms.get_unit(col)
            # Prepare new column name
            new_col = self._col_name_append(col, append=f'{add_col}{bands}')
            # Calculate fit values
            new_values = func_2bnd(df.ms.x, *coefs)
            # Assign values and metadata
            df.ms._set_column_state(new_col, new_values, unit, set_axis, add_label)

        return coefs, self._if_inplace(df, inplace)

    def calculate_twoband(self,
                          p: tuple[float, float, float, float],
                          cols: str | list[str] | None = None,
                          *,
                          kinds: str | list[str],
                          bands: str,
                          append: str = '2bnd',
                          set_axes: str | list[str] | None = None,
                          add_labels: str | list[str] | None = None,
                          inplace: bool = False
                          ) -> pd.DataFrame | None:
        # Default to y axis column if None provided
        cols = self._prepare_values_list(cols, default=self.col_y, func=self.ms.get_column)
        n_cols = len(cols)

        # Prepare other parameters
        # keep existing units
        units = self._prepare_values_list(cols, default='', func=self.ms.get_unit, n=n_cols)
        set_axes = self._prepare_values_list(set_axes, default=None, n=n_cols)
        add_labels = self._prepare_values_list(add_labels, default=None, n=n_cols)

        # Generate new column names
        new_cols = [self._col_name_append(col, append + bands) for col in cols]

        # Prepare twoband functions list
        def kind2func(kind, bands):
            mapping = {'xx': etr.gen_mr2bnd_eq(bands),
                       'xy': etr.gen_hall2bnd_eq(bands)}
            return mapping.get(kind)

        kinds = self._prepare_values_list(kinds, default=None, n=n_cols)
        funcs = self._prepare_values_list(kinds, default=None,
                                          func=lambda x: kind2func(x, bands),
                                          n=n_cols)

        # Work on a copy of the data
        df = self._get_df_copy()

        # Calculate fit values
        new_values = [func(self.x, *p) for func in funcs]

        # Assign values and metadata
        df.ms._set_column_states(
            columns=new_cols,
            values=new_values,
            units=units,
            axes=set_axes,
            labels=add_labels)

        return self._if_inplace(df, inplace)

