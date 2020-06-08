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

from os import getcwd
from os.path import abspath
from time import sleep
from types import MappingProxyType
from functools import partial
from itertools import count
from typing import (
    Optional, Sequence, List, Union, Any, Dict, TypeVar, Mapping,
    overload, TYPE_CHECKING, Tuple, Type
)

import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, DuplicateKeyError

from rdkit.Chem import Mol
from scm.plams import Settings, Molecule
from nanoutils import Literal

from CAT.logger import logger
from CAT.mol_utils import from_rdmol  # noqa: F401
from CAT.workflows import HDF5_INDEX, OPT, MOL
from .create_database import _create_csv, _create_yaml, _create_hdf5, _create_mongodb
from .context_managers import OpenYaml, OpenLig, OpenQD
from .database_functions import (
    df_to_mongo_dict, even_index, from_pdb_array, sanitize_yaml_settings, as_pdb_array
)

if TYPE_CHECKING:
    from os import PathLike  # noqa: F401

__all__ = ['Database']

Ligand = Literal['ligand', 'ligand_no_opt']
QD = Literal['qd', 'qd_no_opt']
ST = TypeVar('ST', bound='Database')


class Database:
    """The Database class.

    .. _pymongo.MongoClient: http://api.mongodb.com/python/current/api/pymongo/mongo_client.html

    Attributes
    ----------
    dirname : :class:`str`
        The path+filename of the directory containing all database components.
    csv_lig : :data:`Callable[..., ContextManager]`
        A function for accesing the context manager for opening
        the .csv file containing all ligand related results.
    csv_qd : :data:`Callable[..., ContextManager]`
        A function for accesing the context manager for opening
        the .csv file containing all quantum dot related results.
    yaml : :data:`Callable[..., ContextManager]`
        A function for accesing the context manager for opening
        the .yaml file containing all job settings.
    hdf5 : :data:`Callable[..., ContextManager]`
        A function for accesing the context manager for opening
        the .hdf5 file containing all structures (as partiallize de-serialized .pdb files).
    mongodb : :class:`Mapping[str, Any]<typing.Mapping>`
        Optional: A dictionary with keyword arguments for pymongo.MongoClient_.
        Defaults to ``None`` if a :exc:`ServerSelectionTimeoutError` is raised when failing to
        contact the host.
        See the **host**, **port** and **kwargs** parameter.

    """  # noqa: E501

    __slots__ = ('__weakref__', '_dirname', '_csv_lig', '_csv_qd', '_yaml',
                 '_hdf5', '_mongodb', '_hash')

    @property
    def dirname(self) -> str:
        """Get the path+filename of the directory containing all database components."""
        return self._dirname

    @property
    def csv_lig(self) -> 'partial[OpenLig]':
        """Get the :attr:`Database.csv_lig` context manager."""
        return self._csv_lig

    @property
    def csv_qd(self) -> 'partial[OpenQD]':
        """Get the :attr:`Database.csv_qd` context manager."""
        return self._csv_qd

    @property
    def yaml(self) -> 'partial[OpenYaml]':
        """Get the :attr:`Database.yaml` context manager."""
        return self._yaml

    @property
    def hdf5(self) -> 'partial[h5py.File]':
        """Get the :attr:`Database.hdf5` context manager."""
        return self._hdf5

    @property
    def mongodb(self) -> Optional[Mapping[str, Any]]:
        """Get the :attr:`Database.mongodb` context manager."""
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
            Optional keyword argument for `pymongo.MongoClient <http://api.mongodb.com/python/current/api/pymongo/mongo_client.html>`_.
            See :attr:`Database.mongodb`.

        """  # noqa: E501
        self._dirname = abspath(path) if path is not None else getcwd()

        # Create the database components and return the filename
        lig_path = _create_csv(self.dirname, database='ligand')
        qd_path = _create_csv(self.dirname, database='qd')
        yaml_path = _create_yaml(self.dirname)
        hdf5_path = _create_hdf5(self.dirname)

        # Populate attributes with MetaManager instances
        self._csv_lig = partial(OpenLig, filename=lig_path)
        self._csv_qd = partial(OpenQD, filename=qd_path)
        self._yaml = partial(OpenYaml, filename=yaml_path)
        self._hdf5 = partial(h5py.File, hdf5_path)

        # Try to create or access the mongodb database
        try:
            self._mongodb = MappingProxyType(_create_mongodb(host, port, **kwargs))
        except ServerSelectionTimeoutError:
            self._mongodb = None

    def __repr__(self) -> str:
        """Return a human string representation of this instance."""
        def _dict_to_str(value: dict) -> str:
            iterator = sorted(value.items(), key=str)
            return '{' + newline.join(f'{repr(k)}: {repr(v)}' for k, v in iterator) + '}'

        def _get_str(key: str, value: Any) -> str:
            func = _dict_to_str if isinstance(value, dict) else repr
            return f'    {key:{offset}} = {func(value)}'

        offset = max(len(k) for k in vars(self))
        newline = ',\n' + ' ' * (6 + offset)

        ret = ',\n'.join(_get_str(k, v) for k, v in vars(self).items())
        return f'Database(\n{ret}\n)'

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if type(self) is not type(value):
            return False

        ret = self.dirname == value.dirname and self.mongodb == value.mongodb
        if not ret:
            return False

        partial_names = ('csv_lig', 'csv_qd', 'yaml', 'hdf5')
        iterator = ((getattr(self, name), getattr(value, name)) for name in partial_names)
        for func1, func2 in iterator:
            ret &= func1.args == func2.args and func1.keywords == func2.keywords and func1.func is func2.func  # noqa: E501
        return ret

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`"""
        try:
            return self._hash
        except AttributeError:
            cls, args, state = self.__reduce__()
            if state is not None:
                state = frozenset(state.items())
            self._hash: int = hash((cls, args, state))
            return self._hash

    def __reduce__(self) -> Tuple[Type[ST], Tuple[str], Optional[Dict[str, Any]]]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        return cls, (self.dirname,), self.mongodb

    def __setstate__(self, state: Optional[Mapping[str, Any]]) -> None:
        """Helper for :mod:`pickle`."""
        if state is None:
            self.mongodb = None
            return

        try:
            self.mongodb = _create_mongodb(**state)
        except ServerSelectionTimeoutError:
            self.mongodb = None

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

    def update_mongodb(self, database: Union[str, Dict[str, pd.DataFrame]] = 'ligand',
                       overwrite: bool = False) -> None:
        """Export ligand or qd results to the MongoDB database.

        Examples
        --------
        .. code:: python

            >>> from CAT import Database

            >>> db = Database(**kwargs)

            # Update from db.csv_lig
            >>> db.update_mongodb('ligand')

            # Update from a lig_df, a user-provided DataFrame
            >>> db.update_mongodb({'ligand': lig_df})
            >>> print(type(lig_df))
            <class 'pandas.core.frame.DataFrame'>

        Parameters
        ----------
        database : |str|_ or |dict|_ [|str|_, |pd.DataFrame|_]
            The type of database.
            Accepted values are ``"ligand"`` and ``"qd"``,
            opening :attr:`Database.csv_lig` and :attr:`Database.csv_qd`, respectivelly.
            Alternativelly, a dictionary with the database name and a matching DataFrame
            can be passed directly.

        overwrite : bool
            Whether or not previous entries can be overwritten or not.

        """
        if self.mongodb is None:
            raise ValueError('Database.Mongodb is None')

        # Open the MongoDB database
        client = MongoClient(**self.mongodb)
        mongo_db = client.cat_database

        if isinstance(database, dict):
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

    def update_csv(self, df: pd.DataFrame,
                   database: Union[Ligand, QD] = 'ligand',
                   columns: Optional[Sequence] = None,
                   overwrite: bool = False,
                   job_recipe: Optional[Settings] = None,
                   status: Optional[str] = None) -> None:
        """Update :attr:`Database.csv_lig` or :attr:`Database.csv_qd` with new settings.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` (:attr:`Database.csv_lig`)
            and ``"qd"`` (:attr:`Database.csv_qd`).

        columns : |Sequence|_
            Optional: A list of column keys in **df** which
            (potentially) are to be added to this instance.
            If ``None``: Add all columns.

        overwrite : |bool|_
            Whether or not previous entries can be overwritten or not.

        job_recipe : |plams.Settings|_
            Optional: A :class:`.Settings` instance with settings specific to a job.

        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.

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
            db.df = even_index(db.df, df)

            # Filter columns
            if not columns:
                df_columns = df.columns
            else:
                df_columns = pd.Index(columns)

            # Update **db.columns**
            bool_ar = df_columns.isin(db.columns)
            for i in df_columns[~bool_ar]:
                if 'job_settings' in i[0]:
                    self._update_hdf5_settings(df, i[0])
                    del df[i]
                    idx = columns.index(i)
                    columns.pop(idx)
                    continue
                try:
                    db[i] = np.array((None), dtype=df[i].dtype)
                except TypeError:  # e.g. if csv[i] consists of the datatype np.int64
                    db[i] = -1

            # Update **self.hdf5**; returns a new series of indices
            hdf5_series = self.update_hdf5(
                df, database=database, overwrite=overwrite, status=status
            )

            # Update **db.values**
            db.update(df[columns], overwrite=overwrite)
            db.update(hdf5_series, overwrite=True)
            df.update(hdf5_series, overwrite=True)
            if status == 'optimized':
                db.update(df[OPT], overwrite=True)

    def update_yaml(self, job_recipe: Settings) -> dict:
        """Update :attr:`Database.yaml` with (potentially) new user provided settings.

        Parameters
        ----------
        job_recipe : |plams.Settings|_
            A settings object with one or more settings specific to a job.

        Returns
        -------
        |dict|_
            A dictionary with the column names as keys and the key for :attr:`Database.yaml`
            as matching values.

        """
        ret = {}
        with self.yaml() as db:
            for item in job_recipe:
                # Unpack and sanitize keys
                key = job_recipe[item].key
                if isinstance(key, type):
                    key = key.__name__

                # Unpack and sanitize values
                value = job_recipe[item].value
                if isinstance(value, dict):
                    value = sanitize_yaml_settings(value, key)

                # Check if the appropiate key is available in **self.yaml**
                if key not in db:
                    db[key] = []

                # Check if the appropiate value is available in **self.yaml**
                if value in db[key]:
                    ret[item] = '{} {:d}'.format(key, db[key].index(value))
                else:
                    db[key].append(value)
                    ret[item] = '{} {:d}'.format(key, len(db[key]) - 1)
        return ret

    def update_hdf5(self, df: pd.DataFrame,
                    database: str = 'ligand',
                    overwrite: bool = False,
                    status: Optional[str] = None):
        """Export molecules (see the ``"mol"`` column in **df**) to the structure database.

        Returns a series with the :attr:`Database.hdf5` indices of all new entries.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.

        overwrite : bool
            Whether or not previous entries can be overwritten or not.

        status : :class:`str`, optional
            A descriptor of the status of the moleculair structures.
            Set to ``"optimized"`` to treat them as optimized geometries.

        Returns
        -------
        |pd.Series|_
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

        # Add new entries to the database
        self.hdf5_availability()
        with self.hdf5('r+') as f:
            i, j = f[database].shape

            if new.any():
                pdb_array = as_pdb_array(df[MOL][new.index], min_size=j)

                # Reshape and update **self.hdf5**
                k = i + pdb_array.shape[0]
                f[database].shape = k, pdb_array.shape[1]
                f[database][i:k] = pdb_array

                ret = pd.Series(np.arange(i, k), index=new.index, name=HDF5_INDEX)
                df.update(ret, overwrite=True)
                if opt:
                    df.loc[new.index, OPT] = True
            else:
                ret = pd.Series(name=HDF5_INDEX, dtype=int)

            # If **overwrite** is *True*
            if overwrite and old.any():
                ar = as_pdb_array(df[MOL][old.index], min_size=j)

                # Ensure that the hdf5 indices are sorted
                idx = np.argsort(old)
                old = old[idx]
                f[database][old] = ar[idx]
                if opt:
                    df.loc[idx.index, OPT] = True

        return ret

    def _update_hdf5_settings(self, df: pd.DataFrame,
                              column: str) -> None:
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
                  ax2: int = 0,
                  ax3: int = 0) -> np.ndarray:
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
    def _get_line_count(filename: str) -> int:
        """Return the total number of lines in **filename**."""
        substract = 0
        with open(filename, 'r') as f:
            for i, j in enumerate(f, 1):
                if j == '\n':
                    substract += 1
        return i - substract

    """ ########################  Pulling results from the database ########################### """

    def from_csv(self, df: pd.DataFrame,
                 database: Union[Ligand, QD] = 'ligand',
                 get_mol: bool = True,
                 inplace: bool = True) -> Optional[pd.Series]:
        """Pull results from :attr:`Database.csv_lig` or :attr:`Database.csv_qd`.

        Performs in inplace update of **df** if **inplace** = ``True``, thus returing ``None``.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.

        get_mol : bool
            Attempt to pull preexisting molecules from the database.
            See the **inplace** argument for more details.

        inplace : bool
            If ``True`` perform an inplace update of the ``"mol"`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        |pd.Series|_ [|plams.Molecule|_]
            Optional: A Series of PLAMS molecules if **get_mol** = ``True``
            and **inplace** = ``False``.

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
                     database: str = 'ligand',
                     inplace: bool = True) -> Optional[pd.Series]:
        """A method which handles the retrieval and subsequent formatting of molecules.

        Called internally by :meth:`.Database.from_csv`.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.

        inplace : bool
            If ``True`` perform an inplace update of the ``("mol", "")`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        |pd.Series|_ [|plams.Molecule|_]
            Optional: A Series of PLAMS molecules if **inplace** is ``False``.

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
            rdmol_list = self.from_hdf5(idx, database=database)
            for mol, rdmol in zip(df.loc[df_slice, MOL], rdmol_list):
                mol.from_rdmol(rdmol)
            ret = None

        # Create and return a new series of PLAMS molecules
        else:
            mol_list = self.from_hdf5(idx, database=database, rdmol=False)
            ret = pd.Series(mol_list, index=df[df_slice].index, name=MOL)

        return ret

    def from_hdf5(self, index: Sequence[int],
                  database: str = 'ligand',
                  rdmol: bool = True) -> List[Union[Molecule, Mol]]:
        """Import structures from the hdf5 database as RDKit or PLAMS molecules.

        Parameters
        ----------
        index : |list|_ [|int|_]
            The indices of the to be retrieved structures.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"qd"``.

        rdmol : bool
            If ``True``, return an RDKit molecule instead of a PLAMS molecule.

        close : bool
            If the database component (:attr:`Database.hdf5`) should be closed afterwards.

        Returns
        -------
        |list|_ [|plams.Molecule|_ or |rdkit.Chem.Mol|_]
            A list of PLAMS or RDKit molecules.

        """
        # Convert **index** to an array if it is a series or dataframe
        if hasattr(index, '__array__'):
            index = np.asarray(index).tolist()

        # Open the database and pull entries
        self.hdf5_availability()
        with self.hdf5('r') as f:
            pdb_array = f[database][index]

        # Return a list of RDKit or PLAMS molecules
        return [from_pdb_array(mol, rdmol=rdmol) for mol in pdb_array]

    def hdf5_availability(self, timeout: float = 5.0,
                          max_attempts: Optional[int] = None) -> None:
        """Check if a .hdf5 file is opened by another process; return once it is not.

        If two processes attempt to simultaneously open a single hdf5 file then
        h5py will raise an :class:`OSError`.

        The purpose of this method is ensure that a .hdf5 file is actually closed,
        thus allowing the :meth:`Database.from_hdf5` method to safely access **filename** without
        the risk of raising an :class:`OSError`.

        Parameters
        ----------
        filename : str
            The path+filename of the hdf5 file.

        timeout : float
            Time timeout, in seconds, between subsequent attempts of opening **filename**.

        max_attempts : int
            Optional: The maximum number attempts for opening **filename**.
            If the maximum number of attempts is exceeded, raise an ``OSError``.

        Raises
        ------
        OSError
            Raised if **max_attempts** is exceded.

        """
        err = (f"h5py.File('{self.hdf5.args[0]}') is currently unavailable; "
               f"repeating attempt in {timeout:1.1f} seconds")
        i = max_attempts or np.inf

        while i:
            try:
                with self.hdf5('r+'):
                    return None  # the .hdf5 file can safely be opened
            except OSError as ex:  # the .hdf5 file cannot be safely opened yet
                logger.warning(err)
                exception = ex
                sleep(timeout)
            i -= 1

        raise exception
