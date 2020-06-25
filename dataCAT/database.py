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
from itertools import count
from typing import (
    Optional, Sequence, List, Union, Any, Dict, TypeVar, Mapping,
    overload, Tuple, Type, Iterable
)

import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError

from rdkit.Chem import Mol
from scm.plams import Settings, Molecule
from nanoutils import PathType, TypedDict
from CAT.mol_utils import from_rdmol  # noqa: F401
from CAT.workflows import HDF5_INDEX, OPT, MOL

from .create_database import (_create_csv, _create_yaml, _create_hdf5, _create_mongodb,
                              QD, Ligand, IDX_DTYPE)
from .context_managers import OpenYaml, OpenLig, OpenQD
from .functions import df_to_mongo_dict, even_index, sanitize_yaml_settings, hdf5_availability
from .pdb_array import PDBContainer
from .hdf5_log import update_hdf5_log

__all__ = ['Database']

KT = TypeVar('KT')
ST = TypeVar('ST', bound='Database')


class JobRecipe(TypedDict):
    """A :class:`~typing.TypedDict` representing the input of :class:`.Database.update_yaml`."""

    key: Union[str, type]
    value: Union[str, Settings]


class Database:
    """The Database class."""

    __slots__ = ('__weakref__', '_dirname', '_csv_lig', '_csv_qd', '_yaml',
                 '_hdf5', '_mongodb', '_hash')

    @property
    def dirname(self) -> str:
        """Get the path+filename of the directory containing all database components."""
        return self._dirname

    @property
    def csv_lig(self) -> 'partial[OpenLig]':
        """:data:`Callable[..., dataCAT.OpenLig]<typing.Callable>`: Get a function for constructing an :class:`dataCAT.OpenLig` context manager."""  # noqa: E501
        return self._csv_lig

    @property
    def csv_qd(self) -> 'partial[OpenQD]':
        """:data:`Callable[..., dataCAT.OpenQD]<typing.Callable>`: Get a function for constructing an :class:`dataCAT.OpenQD` context manager."""  # noqa: E501
        return self._csv_qd

    @property
    def yaml(self) -> 'partial[OpenYaml]':
        """:data:`Callable[..., dataCAT.OpenYaml]<typing.Callable>`: Get a function for constructing an :class:`dataCAT.OpenYaml` context manager."""  # noqa: E501
        return self._yaml

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

        # Create the database components and return the filename
        lig_path = _create_csv(self.dirname, database='ligand')
        qd_path = _create_csv(self.dirname, database='qd')
        yaml_path = _create_yaml(self.dirname)
        hdf5_path = _create_hdf5(self.dirname)

        # Populate attributes with MetaManager instances
        self._csv_lig = partial(OpenLig, filename=lig_path)
        self._csv_qd = partial(OpenQD, filename=qd_path)
        self._yaml = partial(OpenYaml, filename=yaml_path)
        self._hdf5 = partial(h5py.File, hdf5_path, libver='latest')

        # Try to create or access the mongodb database
        try:
            self._mongodb: Optional[Mapping[str, Any]] = MappingProxyType(
                _create_mongodb(host, port, **kwargs)
            )
        except ServerSelectionTimeoutError:
            self._mongodb = None

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        attr_tup = ('dirname', 'csv_lig', 'csv_qd', 'yaml', 'hdf5', 'mongodb')
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

        partial_names = ('csv_lig', 'csv_qd', 'yaml', 'hdf5')
        iterator = ((getattr(self, name), getattr(value, name)) for name in partial_names)
        for func1, func2 in iterator:
            ret &= func1.args == func2.args and func1.keywords == func2.keywords and func1.func is func2.func  # noqa: E501
        return ret

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
            self._mongodb = MappingProxyType(_create_mongodb(**state))
        except ServerSelectionTimeoutError:
            self._mongodb = None

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self: ST, memo: Optional[Dict[int, Any]] = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self

    """ #################################  Updating the database ############################## """

    @overload
    def _parse_database(self, database: Ligand) -> 'partial[OpenLig]':
        ...
    @overload  # noqa: E301
    def _parse_database(self, database: QD) -> 'partial[OpenQD]':
        ...
    def _parse_database(self, database):  # noqa: E301
        """Operate on either the ligand or quantum dot database."""
        if database in ('ligand', 'ligand_no_opt'):
            return self.csv_lig
        elif database in ('qd', 'qd_no_opt'):
            return self.csv_qd
        raise ValueError(f"database={database!r}; accepted values for are 'ligand' and 'qd'")

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
        if self.mongodb is None:
            raise ValueError('Database.Mongodb is None')

        # Open the MongoDB database
        client = MongoClient(**self.mongodb)
        mongo_db = client.cat_database

        if callable(getattr(database, 'items', None)):
            database, db = next(iter(database.items()))  # type: ignore
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

    def update_csv(self, df: pd.DataFrame,
                   database: Union[Ligand, QD] = 'ligand',
                   columns: Optional[Sequence] = None,
                   overwrite: bool = False,
                   job_recipe: Optional[Settings] = None,
                   status: Optional[str] = None) -> None:
        """Update :attr:`Database.csv_lig` or :attr:`Database.csv_qd` with new settings.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` (:attr:`Database.csv_lig`)
            and ``"qd"`` (:attr:`Database.csv_qd`).
        columns : :class:`~collections.abc.Sequence`, optional
            Optional: A sequence of column keys in **df** which
            (potentially) are to be added to this instance.
            If :data:`None` Add all columns.
        overwrite : :class:`bool`
            Whether or not previous entries can be overwritten or not.
        job_recipe : :class:`plams.Settings<scm.plams.core.settings.Settings>`
            Optional: A Settings instance with settings specific to a job.
        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.


        :rtype: :data:`None`

        """
        # Operate on either the ligand or quantum dot database
        manager = self._parse_database(database)

        # Update **self.yaml**
        if job_recipe is not None:
            job_settings = self.update_yaml(job_recipe)
            for key, value in job_settings.items():
                key = ('settings', ) + key
                df[key] = value

        with manager(write=True) as db:
            # Update **db.index**
            db.ndframe = even_index(db.ndframe, df)

            # Filter columns
            if columns is None:
                df_columns = df.columns
            else:
                df_columns = pd.Index(columns)

            # Update **db.columns**
            bool_ar = df_columns.isin(db.columns)
            drop_idx = []
            for i in df_columns[~bool_ar]:
                if 'job_settings' in i[0]:
                    self._update_hdf5_settings(df, i[0])
                    del df[i]
                    drop_idx.append(i)
                    continue
                try:
                    db[i] = np.array((None), dtype=df[i].dtype)
                except TypeError:  # e.g. if csv[i] consists of the datatype np.int64
                    db[i] = -1
            df_columns = df_columns.drop(drop_idx)

            # Update **self.hdf5**; returns a new series of indices
            hdf5_series = self.update_hdf5(
                df, database=database, overwrite=overwrite, status=status
            )

            # Update **db.values**
            db.update(df[df_columns], overwrite=overwrite)
            db.update(hdf5_series, overwrite=True)
            df.update(hdf5_series, overwrite=True)
            if status == 'optimized':
                db.update(df[OPT], overwrite=True)

    def update_yaml(self, job_recipe: Mapping[KT, JobRecipe]) -> Dict[KT, str]:
        """Update :attr:`Database.yaml` with (potentially) new user provided settings.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import Database

            >>> db = Database(...)  # doctest: +SKIP
            >>> job_recipe = {
            ...     'job1': {'key': 'ADFJob', 'value': ...},
            ...     'job2': {'key': 'AMSJob', 'value': ...}
            ... }

            >>> db.update_yaml(job_recipe)  # doctest: +SKIP


        Parameters
        ----------
        job_recipe : :class:`~collections.abc.Mapping`
            A mapping with the settings of one or more jobs.

        Returns
        -------
        :class:`Dict[str, str]<typing.Dict>`
            A dictionary with the column names as keys and the key for :attr:`Database.yaml`
            as matching values.

        """
        ret = {}
        with self.yaml() as db:
            for item, v in job_recipe.items():
                # Unpack and sanitize keys
                key = v['key']
                if isinstance(key, type):
                    key = key.__name__

                # Unpack and sanitize values
                value = v['value']
                if isinstance(value, dict):
                    value = sanitize_yaml_settings(value, key)

                # Check if the appropiate key is available in **self.yaml**
                if key not in db:
                    db[key] = []

                # Check if the appropiate value is available in **self.yaml**
                if value in db[key]:
                    ret[item] = f'{key} {db[key].index(value)}'
                else:
                    db[key].append(value)
                    ret[item] = f'{key} {len(db[key]) - 1}'
        return ret

    def update_hdf5(self, df: pd.DataFrame,
                    database: Union[Ligand, QD] = 'ligand',
                    overwrite: bool = False,
                    status: Optional[str] = None) -> pd.Series:
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
            new = df[HDF5_INDEX][df[OPT] == False] & ~df[MOL].isnull()  # noqa
            old = df[HDF5_INDEX][df[OPT] == True]  # noqa
            opt = True
        else:
            new = df[HDF5_INDEX][df[HDF5_INDEX] == -1] & ~df[MOL].isnull()
            old = df[HDF5_INDEX][df[HDF5_INDEX] >= 0]
            opt = False

        idx_dtype = IDX_DTYPE[database]

        # Add new entries to the database
        self.hdf5_availability()
        with self.hdf5('r+') as f:
            group = f[database]
            if new.any():
                mol_series = df.loc[new.index, MOL]

                index = new.index.values.astype(idx_dtype)
                if database in {'qd', 'qd_no_opt'}:
                    # TODO: Fix the messy MultiIndex
                    core_anchor = index['core anchor']
                    for i, j in enumerate(core_anchor):
                        j_split = j.split()
                        core_anchor[i] = np.fromiter(j_split, count=len(j_split), dtype=np.int32)

                pdb_new = PDBContainer.from_molecules(mol_series, index=index.view(np.recarray))
                pdb_new.to_hdf5(group, mode='append')

                j = len(group['atoms'])
                i = j - len(mol_series)
                ret = pd.Series(np.arange(i, j), index=new.index, name=HDF5_INDEX)

                update_hdf5_log(group, idx=ret.values, message='append')
                df.update(ret, overwrite=True)
                if opt:
                    df.loc[new.index, OPT] = True
            else:
                ret = pd.Series(name=HDF5_INDEX, dtype=int)

            # If **overwrite** is *True*
            if overwrite and old.any():
                old.sort_values(inplace=True)
                mol_series = df.loc[old.index, MOL]

                index = mol_series.index.values.astype(idx_dtype).view(np.recarray)
                pdb_old = PDBContainer.from_molecules(mol_series, index=index)
                pdb_old.to_hdf5(group, mode='update', idx=old.values)
                update_hdf5_log(group, idx=old.values, message='update')
                if opt:
                    df.loc[old.index, OPT] = True
        return ret

    def _update_hdf5_settings(self, df: pd.DataFrame, column: str) -> None:
        """Export all files in **df[column]** to hdf5 dataset **column**."""
        # Add new entries to the database
        self.hdf5_availability()
        with self.hdf5('r+') as f:
            i, j, k = f[column].shape

            # Create a 3D array of input files
            try:
                job_ar = self._read_inp(df[column], j, k)
            except ValueError:  # df[column] consists of empty lists, abort
                return None

            # Reshape **self.hdf5**
            k = max(i, 1 + int(df[HDF5_INDEX].max()))
            f[column].shape = k, job_ar.shape[1], job_ar.shape[2]

            # Update the hdf5 dataset
            idx = df[HDF5_INDEX].astype(int, copy=False)
            idx_argsort = np.argsort(idx)
            f[column][idx[idx_argsort]] = job_ar[idx_argsort]
        return None

    @staticmethod
    def _read_inp(job_paths: Sequence[str],
                  ax2: int = 0, ax3: int = 0) -> np.ndarray:
        """Convert all files in **job_paths** (nested sequence of filenames) into a 3D array."""
        # Determine the minimum size of the to-be returned 3D array
        line_count = [[Database._get_line_count(j) for j in i] for i in job_paths]
        ax1 = len(line_count)
        ax2 = max(ax2, max(len(i) for i in line_count))
        ax3 = max(ax3, max(j for i in line_count for j in i))

        # Create and return a padded 3D array of strings
        ret = np.zeros((ax1, ax2, ax3), dtype='S120')
        for i, list1, list2 in zip(count(), line_count, job_paths):
            for j, k, filename in zip(count(), list1, list2):
                ret[i, j, :k] = np.loadtxt(filename, dtype='S120', comments=None, delimiter='\n')
        return ret

    @staticmethod
    def _get_line_count(filename: PathType) -> int:
        """Return the total number of lines in **filename**."""
        substract = 0
        with open(filename, 'r') as f:
            for i, j in enumerate(f, 1):
                if j == '\n':
                    substract += 1
        return i - substract

    """ ########################  Pulling results from the database ########################### """

    def from_csv(self, df: pd.DataFrame, database: Union[Ligand, QD] = 'ligand',
                 get_mol: bool = True, inplace: bool = True) -> Optional[pd.Series]:
        """Pull results from :attr:`Database.csv_lig` or :attr:`Database.csv_qd`.

        Performs in inplace update of **df** if **inplace** = :data:`True`,
        thus returing :data:`None`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        get_mol : :class:`bool`
            Attempt to pull preexisting molecules from the database.
            See the **inplace** argument for more details.
        inplace : :class:`bool`
            If :data:`True` perform an inplace update of the ``"mol"`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        :class:`pandas.Series`, optional
            Optional: A Series of PLAMS molecules if **get_mol** = :data:`True`
            and **inplace** = :data:`False`.

        """
        # Operate on either the ligand or quantum dot database
        manager = self._parse_database(database)

        # Update the *hdf5 index* column in **df**
        with manager(write=False) as db:
            df.update(db.ndframe, overwrite=True)
            df[HDF5_INDEX] = df[HDF5_INDEX].astype(int, copy=False)

        # **df** has been updated and **get_mol** = *False*
        if not get_mol:
            return None
        return self._get_csv_mol(df, database, inplace)

    def _get_csv_mol(self, df: pd.DataFrame,
                     database: Union[Ligand, QD] = 'ligand',
                     inplace: bool = True) -> Optional[pd.Series]:
        """A method which handles the retrieval and subsequent formatting of molecules.

        Called internally by :meth:`Database.from_csv`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A dataframe of new (potential) database entries.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        inplace : :class:`bool`
            If :data:`True` perform an inplace update of the ``("mol", "")`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        :class:`pandas.Series`, optional
            Optional: A Series of PLAMS molecules if **inplace** is :data:`False`.

        """
        # Sort and find all valid HDF5 indices
        df.sort_values(by=[HDF5_INDEX], inplace=True)
        if 'no_opt' in database:
            df_slice = df[HDF5_INDEX] >= 0
        else:
            df_slice = df[OPT] == True  # noqa
        idx = df[HDF5_INDEX][df_slice].values

        # If no HDF5 indices are availble in **df** then abort the function
        if not df_slice.any():
            if inplace:
                return None
            return pd.Series(None, name=MOL, dtype=object)

        # Update **df** with preexisting molecules from **self**, returning *None*
        if inplace:
            self.from_hdf5(idx, database=database, mol_list=df.loc[df_slice, MOL], rdmol=False)
            return None

        # Create and return a new series of PLAMS molecules
        else:
            mol_list = self.from_hdf5(idx, database=database, rdmol=False)
            return pd.Series(mol_list, index=df[df_slice].index, name=MOL)

    def from_hdf5(self, index: Union[slice, Sequence[int]],
                  database: Union[Ligand, QD] = 'ligand', rdmol: bool = True,
                  mol_list: Optional[Iterable[Molecule]] = None) -> List[Union[Molecule, Mol]]:
        """Import structures from the hdf5 database as RDKit or PLAMS molecules.

        Parameters
        ----------
        index : :class:`Sequence[int]<typing.Sequence>` or :class:`slice`
            The indices of the to be retrieved structures.
        database : :class:`str`
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.
        rdmol : :class:`bool`
            If :data:`True`, return an RDKit molecule instead of a PLAMS molecule.

        Returns
        -------
        :class:`List[plams.Molecule]<typing.List>` or :class:`List[rdkit.Mol]<typing.List>`
            A list of PLAMS or RDKit molecules.

        """
        # Open the database and pull entries
        self.hdf5_availability()
        with self.hdf5('r') as f:
            pdb = PDBContainer.from_hdf5(f[database], index)
            mol_list_ = pdb.to_molecules(mol=mol_list)

        if rdmol:
            return [from_rdmol(mol) for mol in mol_list_]
        return mol_list_

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
