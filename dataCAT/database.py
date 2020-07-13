"""A module which holds the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    Database

API
---
.. autoclass:: Database
    :members:

"""

import reprlib
import textwrap
from os import getcwd, PathLike
from os.path import abspath
from types import MappingProxyType
from functools import partial
from typing import (
    Optional, Union, Any, Dict, TypeVar, Mapping, FrozenSet,
    overload, Tuple, Type, Iterable, ClassVar, TYPE_CHECKING
)

import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError

from CAT.workflows import HDF5_INDEX, OPT, MOL

from .create_database import create_hdf5, create_mongodb, Name
from .functions import df_to_mongo_dict, hdf5_availability
from .pdb_array import PDBContainer
from .hdf5_log import update_hdf5_log
from .property_dset import create_prop_dset, update_prop_dset
from ._parse_settings import _update_hdf5_settings
from ._read_database import df_from_hdf5

if TYPE_CHECKING:
    from numpy.typing import DtypeLike, ArrayLike
else:
    DtypeLike = 'numpy.typing.DtypeLike'
    ArrayLike = 'numpy.typing.ArrayLike'

__all__ = ['Database']

KT = TypeVar('KT')
ST = TypeVar('ST', bound='Database')
MIT = TypeVar('MIT', bound=pd.MultiIndex)


def _filter(ar: ArrayLike) -> np.ndarray:
    return ~np.asarray(ar, dtype=bool)


class Database:
    """The Database class."""

    __slots__ = ('__weakref__', '_dirname', '_hdf5', '_mongodb', '_hash')

    PORPERTY_BLACKLIST: ClassVar[FrozenSet[str]] = frozenset({
        MOL[0], OPT[0], HDF5_INDEX[0], 'settings'
    })

    @property
    def dirname(self) -> str:
        """Get the path+filename of the directory containing all database components."""
        return self._dirname

    @property
    def hdf5(self) -> 'partial[h5py.File]':
        """:data:`Callable[..., h5py.File]<typing.Callable>`: Get a function for constructing a :class:`h5py.File` context manager."""  # noqa: E501
        return self._hdf5

    @property
    def mongodb(self) -> Optional[Mapping[str, Any]]:
        """:class:`Mapping[str, Any]<typing.Mapping>`, optional: Get a mapping with keyword arguments for :class:`pymongo.MongoClient<pymongo.mongo_client.MongoClient>`."""  # noqa: E501
        return self._mongodb

    def __init__(self, path: Union[str, 'PathLike[str]', None] = None,
                 host: str = 'localhost',
                 port: int = 27017,
                 **kwargs) -> None:
        """Initialize :class:`Database`.

        Parameters
        ----------
        path : str
            The path+directory name of the directory which is to contain all database components
            (see :attr:`Database.dirname`).
        host : str
            Hostname or IP address or Unix domain socket path of a single mongod or
            mongos instance to connect to, or a mongodb URI, or a list of hostnames mongodb URIs.
            If **host** is an IPv6 literal it must be enclosed in ``"["`` and ``"]"`` characters
            following the RFC2732 URL syntax (e.g. ``"[::1]"`` for localhost).
            Multihomed and round robin DNS addresses are not supported.
            See :attr:`Database.mongodb`.
        port : str
            port number on which to connect.
            See :attr:`Database.mongodb`.
        **kwargs
            Optional keyword argument for :class:`pymongo.MongoClient<pymongo.mongo_client.MongoClient>`.
            See :attr:`Database.mongodb`.

        """  # noqa: E501
        self._dirname: str = abspath(path) if path is not None else getcwd()

        # Populate attributes with MetaManager instances
        hdf5_path = create_hdf5(self.dirname)
        self._hdf5 = partial(h5py.File, hdf5_path, libver='latest')

        # Try to create or access the mongodb database
        try:
            self._mongodb: Optional[Mapping[str, Any]] = MappingProxyType(
                create_mongodb(host, port, **kwargs)
            )
        except ServerSelectionTimeoutError:
            self._mongodb = None

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        attr_tup = ('dirname', 'hdf5', 'mongodb')
        attr_max = max(len(i) for i in attr_tup)

        iterator = ((name, getattr(self, name)) for name in attr_tup[:-1])
        args = ',\n'.join(f'{name:{attr_max}} = {attr!r}' for name, attr in iterator)
        args += f',\n{attr_tup[-1]:{attr_max}} = {reprlib.repr(self.mongodb)}'

        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(args, indent)}\n)'

    def __eq__(self, value: Any) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False

        ret: bool = self.dirname == value.dirname and self.mongodb == value.mongodb
        if not ret:
            return False

        self_hdf5 = self.hdf5
        value_hdf5 = value.hdf5

        return (self_hdf5.args == value_hdf5.args and
                self_hdf5.keywords == value_hdf5.keywords and
                self_hdf5.func is value_hdf5.func)

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            cls, args, state = self.__reduce__()
            if state is not None:
                state = frozenset(state.items())  # type: ignore
            self._hash: int = hash((cls, args, state))
            return self._hash

    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[str], Optional[Dict[str, Any]]]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        mongodb = self.mongodb if self.mongodb is None else dict(self.mongodb)
        return cls, (self.dirname,), mongodb

    def __setstate__(self, state: Optional[Dict[str, Any]]) -> None:
        """Helper for :mod:`pickle` and :meth:`~Database.__reduce__`."""
        if state is None:
            self._mongodb = None
            return

        try:
            self._mongodb = MappingProxyType(create_mongodb(**state))
        except ServerSelectionTimeoutError:
            self._mongodb = None

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self: ST, memo: Optional[Dict[int, Any]] = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self

    """ #################################  Updating the database ############################## """

    def update_mongodb(self, database: Union[str, Mapping[str, pd.DataFrame]] = 'ligand',
                       overwrite: bool = False) -> None:
        """Export ligand or qd results to the MongoDB database.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import Database

            >>> kwargs = dict(...)  # doctest: +SKIP
            >>> db = Database(**kwargs)  # doctest: +SKIP

            # Update from db.csv_lig
            >>> db.update_mongodb('ligand')  # doctest: +SKIP

            # Update from a lig_df, a user-provided DataFrame
            >>> db.update_mongodb({'ligand': lig_df})  # doctest: +SKIP
            >>> print(type(lig_df))  # doctest: +SKIP
            <class 'pandas.core.frame.DataFrame'>

        Parameters
        ----------
        database : :class:`str` or :class:`Mapping[str, pandas.DataFrame]<typing.Mapping>`
            The type of database.
            Accepted values are ``"ligand"`` and ``"qd"``,
            opening :attr:`Database.csv_lig` and :attr:`Database.csv_qd`, respectivelly.
            Alternativelly, a dictionary with the database name and a matching DataFrame
            can be passed directly.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.


        :rtype: :data:`None`

        """
        raise NotImplementedError

        if self.mongodb is None:
            raise ValueError('Database.Mongodb is None')

        # Open the MongoDB database
        client = MongoClient(**self.mongodb)
        mongo_db = client.cat_database

        if callable(getattr(database, 'items', None)):
            database, db = next(iter(database.items()))
            dict_gen = df_to_mongo_dict(db)
            idx_keys = db.index.names
            collection = mongo_db.ligand_database if database == 'ligand' else mongo_db.qd_database
        else:
            # Operate on either the ligand or quantum dot database
            if database == 'ligand':
                idx_keys = ('smiles', 'anchor')
                collection = mongo_db.ligand_database
                manager = self.csv_lig
            elif database == 'qd':
                idx_keys = ('core', 'core anchor', 'ligand smiles', 'ligand anchor')
                collection = mongo_db.qd_database
                manager = self.csv_lig

            # Parse the ligand or qd dataframe
            with manager(write=False) as db:
                dict_gen = df_to_mongo_dict(db)

        # Update the collection
        # Try to insert al keys at once
        try:
            collection.insert_many(dict_gen)
        except DuplicateKeyError:
            pass
        else:
            return

        # Simultaneous insertion failed, resort to plan B
        for item in dict_gen:
            try:
                collection.insert_one(item)
            except DuplicateKeyError:  # An item is already present in the collection
                if overwrite:
                    filter_ = {i: item[i] for i in idx_keys}
                    collection.replace_one(filter_, item)

    def from_df(self, df: pd.DataFrame, df_bool: pd.DataFrame, name: Name,
                columns: Optional[ArrayLike] = None,
                overwrite: bool = False,
                status: Optional[str] = None,
                ) -> None:
        """Update :attr:`Database.csv_lig` or :attr:`Database.csv_qd` with new settings.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        name : :class:`str`
            The type of database; accepted values are ``"ligand"`` (:attr:`Database.csv_lig`)
            and ``"qd"`` (:attr:`Database.csv_qd`).
        columns : :class:`~collections.abc.Sequence`, optional
            Optional: A sequence of column keys in **df** which
            (potentially) are to be added to this instance.
            If :data:`None` Add all columns.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.
        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.


        :rtype: :data:`None`

        """
        if isinstance(columns, pd.MultiIndex):
            df_columns: pd.MultiIndex = columns
        else:
            df_columns = df.columns if columns is None else pd.Index(columns)
        if len(getattr(df_columns, 'levels', ())) != 2:
            raise ValueError("'columns' expected a 2-level MultiIndex")

        self.hdf5_availability()
        with self.hdf5('r+') as f:
            # Update molecules
            grp = f[name]
            if MOL in df_columns:
                self.update_hdf5(df, df_bool, grp, overwrite=overwrite, status=status)

            # Update properties
            prop_grp = grp['properties']
            self._update_properties(prop_grp, df, df_bool, df_columns, overwrite=overwrite)

            # Update the job settings
            settings = (j for i, j in df_columns if i == 'settings')
            self._update_job_settings(f, df, settings)

    @classmethod
    def _update_properties(cls, group: h5py.Group, df: pd.DataFrame, df_bool: pd.DataFrame,
                           columns: pd.MultiIndex, overwrite: bool = False) -> None:
        # Identify the property-containing columns
        lvl0 = set(columns.levels[0]).difference(cls.PORPERTY_BLACKLIST)
        column_iterator = ((k, columns.get_loc_level(k)[1]) for k in lvl0)

        parent = group.parent
        for n, name_seq in column_iterator:
            index = df_bool[n]
            if isinstance(index, pd.DataFrame):
                index = index.any(axis=1)
            data = df.loc[index, n].values
            hdf5_index = df.loc[index, HDF5_INDEX].values

            # Get (or set) the dataset
            try:
                dset = group[n]
            except KeyError:
                if len(name_seq) == 1:
                    name_seq = None
                dset = create_prop_dset(group, n, data.dtype, name_seq)

            # Update the dataset
            update_prop_dset(dset, data, hdf5_index)

            # Add an entry to the logger
            message = f'datasets={[dset.name]!r}; overwrite={overwrite!r}'
            update_hdf5_log(parent['logger'], hdf5_index, message=message)

    def _update_job_settings(self, f: h5py.File, df: pd.DataFrame, columns: Iterable[str]) -> None:
        """Even the columns of **df** and **db**."""
        column_set = set(columns)
        while column_set:
            name = column_set.pop()
            _update_hdf5_settings(f, df, name)
            del df['settings', name]

    def update_hdf5(self, df: pd.DataFrame, df_bool: pd.DataFrame,
                    group: h5py.Group,
                    overwrite: bool = False,
                    status: Optional[str] = None) -> None:
        """Export molecules (see the ``"mol"`` column in **df**) to the structure database.

        Returns a series with the :attr:`Database.hdf5` indices of all new entries.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.
        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.

        Returns
        -------
        :class:`pandas.Series`
            A series with the indices of all new molecules in :attr:`Database.hdf5`.

        """
        # Identify new and preexisting entries
        if status == 'optimized':
            index = df_bool[OPT]
            is_opt = True
        else:
            index = df_bool[MOL]
            is_opt = False

        new = df.loc[index, HDF5_INDEX]
        old = df.loc[~index, HDF5_INDEX]

        if new.size:
            self._write_hdf5(group, df, new, opt=is_opt, overwrite=False)

        # If **overwrite** is *True*
        if overwrite and old.size:
            self._write_hdf5(group, df, old, opt=is_opt, overwrite=True)

    @classmethod
    def _write_hdf5(cls, group: h5py.Group, df: pd.DataFrame, hdf5_series: pd.Series,
                    opt: bool = False, overwrite: bool = False) -> pd.Index:
        """Helper method for :meth:`update_hdf5` when :code:`overwrite = False`."""
        index = hdf5_series.index
        hdf5_index = hdf5_series.values
        mol_series = df.loc[index, MOL]

        # Export the molecules to the .hdf5 file
        dtype: np.dtype = group['index'].dtype
        scale: np.ndarray = index.values.astype(dtype, copy=False)
        pdb_new = PDBContainer.from_molecules(mol_series, scale=scale)
        pdb_new.to_hdf5(group, index=hdf5_index, update_scale=not opt)

        # Post a message in the logger
        names = PDBContainer.DSET_NAMES.values()
        message = f"datasets={[group[n].name for n in names]!r}; overwrite={overwrite}"
        update_hdf5_log(group['logger'], index=hdf5_index, message=message)

        if opt:
            df.loc[index, OPT] = True

    """ ########################  Pulling results from the database ########################### """

    @overload
    def to_df(self, index: Optional[ArrayLike], name: Name, *properties: str,
              read_mol: bool = True) -> pd.DataFrame:
        ...
    @overload
    def to_df(self, df: pd.DataFrame, name: Name, *prop_names: str,
              read_mol: bool = True) -> pd.DataFrame:
        ...
    def to_df(self, arg, name, *prop_names, read_mol=True):
        r"""Construct or update a dataframe with the specified data.

        Parameters
        ----------
        index : array-like
            An array-like object representing an index; used for constructing a new DataFrame.
            Serves as an alternative to **df**.
        df : :class:`pandas.DataFrame`
            A to-be updated DataFrame.
            Serves as an alternative to **index**.
        name : :class:`str`
            The name of the database Group.
            Accepted values are ``"ligand"`` and ``"qd"``,
        \*prop_names : :class:`str`
            The names of 0 or more to-be read properties.
        read_mol : :class:`bool`
            Whether or not to read molecules from the database.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame constructed from the database.
            Note that if :code:`df = pd.DataFrame(..)` then the passed DataFrame
            will be returned.

        """
        # Update the *hdf5 index* column in **df**
        with self.hdf5('r+') as f:
            # Prepare the group and datasets
            mol_group = f[name]

            prop_dsets = []
            for i in prop_names:
                try:
                    prop_dsets.append(mol_group[f'properties/{i}'])
                except KeyError:
                    pass

            # Parse **arg**
            dtype = mol_group['index'].dtype
            if isinstance(arg, pd.DataFrame):
                df = arg
                index: Union[slice, np.ndarray] = df.index.values.astype(dtype, copy=False)
                mol_list = df.get(MOL, None)
            else:
                df = None
                index = np.asarray(arg, dtype=dtype) if arg is not None else slice(None)
                mol_list = None

            # Construct the new dataframe
            df_new = df_from_hdf5(
                mol_group, index, *prop_dsets, mol_list=mol_list, read_mol=read_mol
            )
            df_bool = pd.DataFrame(index=df_new.index, columns=df_new.columns, dtype=bool)
            df_bool[:] = ~df_new.values.astype(bool)
            df_bool.columns.names = df.columns.names

            # Define the to-be returned DataFrame
            if df is None:
                ret = df_new
            else:
                df.update(df_new, filter_func=_filter)
                ret = df

            # Sort the index and return
            if HDF5_INDEX in df_bool.columns:
                ret.sort_values([HDF5_INDEX], inplace=True)
                df_bool[HDF5_INDEX] = ret[HDF5_INDEX]
                df_bool.sort_values([HDF5_INDEX], inplace=True)
                df_bool[HDF5_INDEX] = True
            return ret, df_bool

    def hdf5_availability(self, timeout: float = 5.0,
                          max_attempts: Optional[int] = 10) -> None:
        """Check if a .hdf5 file is opened by another process; return once it is not.

        If two processes attempt to simultaneously open a single hdf5 file then
        h5py will raise an :exc:`OSError`.

        The purpose of this method is ensure that a .hdf5 file is actually closed,
        thus allowing the :meth:`Database.from_hdf5` method to safely access **filename** without
        the risk of raising an :exc:`OSError`.

        Parameters
        ----------
        timeout : :class:`float`
            Time timeout, in seconds, between subsequent attempts of opening **filename**.
        max_attempts : :class:`int`, optional
            Optional: The maximum number attempts for opening **filename**.
            If the maximum number of attempts is exceeded, raise an :exc:`OSError`.
            Setting this value to :data:`None` will set the number of attempts to unlimited.

        Raises
        ------
        :exc:`OSError`
            Raised if **max_attempts** is exceded.

        See Also
        --------
        :func:`dataCAT.functions.hdf5_availability`
            This method as a function.

        """
        filename = self.hdf5.args[0]
        hdf5_availability(filename, timeout, max_attempts, libver='latest')
