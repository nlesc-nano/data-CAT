"""A module which holds the Database class."""

__all__ = ['Database']

from os import getcwd
from time import sleep
from typing import (Optional, Sequence, List, Union)
from itertools import count

import yaml
import h5py
import numpy as np
import pandas as pd

from rdkit.Chem import Mol
from scm.plams import Settings, Molecule

from .database_functions import (
    _create_csv, _create_yaml, _create_hdf5, even_index,
    from_pdb_array, sanitize_yaml_settings, as_pdb_array
)
from .utils import from_rdmol


class Database():
    """The Database class.

    Paramaters
    ----------
    path : |str|_
        The path+directory name of the directory which is to contain all database components.

    Attributes
    ----------
    csv_lig : |str|_
        Path+filename of the .csv file containing all ligand related results.

    csv_qd : |str|_
        Path+filename of the .csv file containing all quantum dot related results.

    yaml : |str|_
        Path and filename of the .yaml file containing all job settings.

    hdf5 : |str|_
        Path and filename of the .hdf5 file containing all structures
        (as partiallize de-serialized .pdb files).

    mongodb : |None|_
        Placeholder.

    """

    def __init__(self, path: str = None) -> None:
        path = path or getcwd()

        # Attributes which hold the absolute paths to various components of the database
        self.csv_lig = _create_csv(path, database='ligand')
        self.csv_qd = _create_csv(path, database='QD')
        self.yaml = _create_yaml(path)
        self.hdf5 = _create_hdf5(path)
        self.mongodb = None  # Placeholder

    def __str__(self) -> str:
        ret = Settings()
        attr_dict = vars(self)
        for key in attr_dict:
            ret[key] = type(attr_dict[key])
        return str(ret)

    """ ###########################  Opening and closing the database ######################### """

    class open_yaml():
        """Context manager for opening and closing the job settings database (:attr:`.Database.yaml`).

        Paramaters
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.yaml`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        Attributes
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.yaml`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        settings: |None|_ or |Settings|_
            An attribute for (temporary) storing the opened .yaml file
            (:attr:`.filename`) as :class:`.Settings` instance.

        """

        def __init__(self, filename: Optional[str] = None,
                     write: bool = True) -> None:
            self.filename = filename or getcwd()
            self.write = write
            self.settings = None

        def __enter__(self) -> Settings:
            with open(self.filename, 'r') as f:
                self.settings = Settings(yaml.load(f, Loader=yaml.FullLoader))
                return self.settings

        def __exit__(self, *args) -> None:
            if self.write:
                yml_dict = self.settings.as_dict()

                # A fix for Settings.as_dict() not functioning when containg a lists of Settings
                for key in yml_dict:
                    for i, value in enumerate(yml_dict[key]):
                        if isinstance(value, Settings):
                            yml_dict[key][i] = value.as_dict()

                # Write to the .yaml file
                with open(self.filename, 'w') as f:
                    f.write(yaml.dump(yml_dict, default_flow_style=False, indent=4))
            self.settings = False

    class open_csv_lig():
        """Context manager for opening and closing the ligand database.

        Paramaters
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.csv_lig`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        Attributes
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.csv_lig`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        df: |None|_ or |pd.DataFrame|_
            An attribute for (temporary) storing the opened .csv file
            (:attr:`.filename`) as :class:`.DataFrame` instance.

        """

        def __init__(self, path=None, write=True):
            self.path = path or getcwd()
            self.write = write
            self.df = None

        def __enter__(self):
            # Open the .csv file
            dtype = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
            self.df = Database.DF(
                pd.read_csv(self.path, index_col=[0, 1], header=[0, 1], dtype=dtype)
            )

            # Fix the columns
            idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in self.df.columns]
            columns = pd.MultiIndex.from_tuples(idx_tups, names=self.df.columns.names)
            self.df.columns = columns
            return self.df

        def __exit__(self, type, value, traceback):
            if self.write:
                self.df.to_csv(self.path)
            self.df = None

    class open_csv_qd():
        """Context manager for opening and closing the quantum dot database.

        Paramaters
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.csv_qd`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        Attributes
        ----------
        filename: |str|_
            The path+filename to the database component (:attr:`.Database.csv_qd`).

        write: |bool|_
            Whether or not the database file should be updated after closing this instance.

        df: |None|_ or |pd.DataFrame|_
            An attribute for (temporary) storing the opened .csv file
            (:attr:`.filename`) as :class:`.DataFrame` instance.

        """

        def __init__(self, path=None, write=True):
            self.path = path or getcwd()
            self.write = write
            self.df = None

        def __enter__(self):
            # Open the .csv file
            dtype = {'hdf5 index': int, 'settings': str, 'opt': bool}
            self.df = Database.DF(
                pd.read_csv(self.path, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype)
            )

            # Fix the columns
            idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in self.df.columns]
            columns = pd.MultiIndex.from_tuples(idx_tups, names=self.df.columns.names)
            self.df.columns = columns
            return self.df

        def __exit__(self, type, value, traceback):
            if self.write:
                self.df.to_csv(self.path)
            self.df = None

    class DF(dict):
        """A mutable container for holding dataframes.

        A subclass of :class:`dict` containing a single key (``"df"``) and value
        (a Pandas DataFrame).
        Calling an item or attribute of :class:`.DF` will call said method on the
        underlaying DataFrame (``self["df"]``).
        An exception to this is the ``"df"`` key, which will get/set the DataFrame
        instead.

        """

        def __init__(self, df: pd.DataFrame) -> None:
            super().__init__()
            super().__setitem__('df', df)

        def __getattribute__(self, key):
            if key == 'update_df' or (key.startswith('__') and key.endswith('__')):
                return super().__getattribute__(key)
            return self['df'].__getattribute__(key)

        def __setattr__(self, key, value):
            self['df'].__setattr__(key, value)

        def __setitem__(self, key, value):
            if key == 'df' and not isinstance(value, pd.DataFrame):
                try:
                    value = value['df']
                    if not isinstance(value, pd.DataFrame):
                        raise KeyError
                    super().__setitem__('df', value)
                except KeyError:
                    err = ("Instance of 'pandas.DataFrame' or 'CAT.Database.DF' expected;"
                           " observed type: '{}'")
                    raise TypeError(err.format(value.__class__.__name__))
            elif key == 'df':
                super().__setitem__('df', value)
            else:
                self['df'].__setitem__(key, value)

        def __getitem__(self, key):
            df = super().__getitem__('df')
            if isinstance(key, str) and key == 'df':
                return df
            return df.__getitem__(key)

    """ #################################  Updating the database ############################## """

    def update_csv(self, df: pd.DataFrame,
                   database: str = 'ligand',
                   columns: Optional[Sequence] = None,
                   overwrite: bool = False,
                   job_recipe: Optional[Settings] = None,
                   opt: bool = False) -> None:
        """Update :attr:`.csv_lig` or :attr:`.csv_qd` with new user-provided settings.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` (:attr:`.csv_lig`)
            and ``"QD"`` (:attr:`.csv_qd`).

        columns : |Sequence|_
            Optional: A list of column keys in **df** which
            (potentially) are to be added to this instance.
            If ``None``: Add all columns.

        overwrite : |bool|_
            Whether or not previous entries can be overwritten or not.

        job_recipe : |plams.Settings|_
            Optional: A :class:`.Settings` instance with settings specific to a job.

        opt : |bool|_
            WiP.

        """
        # Operate on either the ligand or quantum dot database
        if database in ('ligand', 'ligand_no_opt'):
            path = self.csv_lig
            open_csv = self.open_csv_lig
        elif database in ('QD', 'QD_no_opt'):
            path = self.csv_qd
            open_csv = self.open_csv_qd

        # Update **self.yaml**
        if job_recipe is not None:
            job_settings = self.update_yaml(job_recipe)
            for key, value in job_settings.items():
                df[('settings', key)] = value

        with open_csv(path, write=True) as db:
            # Update **db.index**
            db['df'] = even_index(db['df'], df)

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
            hdf5_series = self.update_hdf5(df, database=database, overwrite=overwrite, opt=opt)

            # Update **db.values**
            db.update(df[columns], overwrite=overwrite)
            db.update(hdf5_series, overwrite=True)
            if opt:
                db.update(df[('opt', '')], overwrite=True)

    def update_yaml(self, job_recipe: Settings) -> dict:
        """Update :attr:`.yaml` with (potentially) new user provided settings.

        Paramaters
        ----------
        job_recipe : |plams.Settings|_
            A settings object with one or more settings specific to a job.

        Returns
        -------
        |dict|_
            A dictionary with the column names as keys and the key for :attr:`.yaml`
            as matching values.

        """
        ret = {}
        with self.open_yaml(self.yaml) as db:
            for item in job_recipe:
                # Unpack and sanitize keys
                key = job_recipe[item].key
                if isinstance(key, type):
                    key = str(key).rsplit("'", 1)[0].rsplit('.', 1)[-1]

                # Unpack and sanitize values
                value = job_recipe[item].value
                if isinstance(value, dict):
                    value = sanitize_yaml_settings(value, key)

                # Check if the appropiate key is available in **self.yaml**
                if key not in db:
                    db[key] = []

                # Check if the appropiate value is available in **self.yaml**
                if value in db[key]:
                    ret[item] = key + ' ' + str(db[key].index(value))
                else:
                    db[key].append(value)
                    ret[item] = key + ' ' + str(len(db[key]) - 1)
        return ret

    def update_hdf5(self, df: pd.DataFrame,
                    database: str = 'ligand',
                    overwrite: bool = False,
                    opt: bool = False):
        """ Export molecules (see the ``"mol"`` column in **df**) to the structure database.

        Returns a series with the :attr:`.hdf5` indices of all new entries.

        Paramaters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"QD"``.

        overwrite : bool
            Whether or not previous entries can be overwritten or not.

        Returns
        -------
        |pd.Series|_
            A series with the indices of all new molecules in :attr:`.hdf5`.

        """
        # Identify new and preexisting entries
        if opt:
            new = df['hdf5 index'][df['opt'] == False]  # noqa
            old = df['hdf5 index'][df['opt'] == True]  # noqa
        else:
            new = df['hdf5 index'][df['hdf5 index'] == -1]
            old = df['hdf5 index'][df['hdf5 index'] >= 0]

        # Add new entries to the database
        self.hdf5_availability()
        with h5py.File(self.hdf5, 'r+') as f:
            i, j = f[database].shape

            if new.any():
                pdb_array = as_pdb_array(df['mol'][new.index], min_size=j)

                # Reshape and update **self.hdf5**
                k = i + pdb_array.shape[0]
                f[database].shape = k, pdb_array.shape[1]
                f[database][i:k] = pdb_array

                ret = pd.Series(np.arange(i, k), index=new.index, name=('hdf5 index', ''))
                df.update(ret, overwrite=True)
                if opt:
                    df.loc[new.index, ('opt', '')] = True
            else:
                ret = pd.Series(name=('hdf5 index', ''), dtype=int)

            # If **overwrite** is *True*
            if overwrite and old.any():
                ar = as_pdb_array(df['mol'][old.index], min_size=j)

                # Ensure that the hdf5 indices are sorted
                # import pdb; pdb.set_trace()
                idx = np.argsort(old)
                old = old[idx]
                f[database][old] = ar[idx]
                if opt:
                    df.loc[idx.index, ('opt', '')] = True

        return ret

    def _update_hdf5_settings(self, df: pd.DataFrame,
                              column) -> None:
        # Add new entries to the database
        self.hdf5_availability()
        with h5py.File(self.hdf5, 'r+') as f:
            i, j, k = f[column].shape

            # Create a 3D array of input files
            try:
                job_ar = self._read_inp(df[column], j, k)
            except ValueError:  # df[column] consists of empty lists
                return None

            # Reshape **self.hdf5**
            k = max(i, 1 + int(df['hdf5 index'].max()))
            f[column].shape = k, job_ar.shape[1], job_ar.shape[2]

            # Update the hdf5 dataset
            idx = df['hdf5 index'].astype(int, copy=False)
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

    def from_csv(self,
                 df: pd.DataFrame,
                 database: str = 'ligand',
                 get_mol: bool = True,
                 inplace: bool = True) -> Optional[pd.Series]:
        """Pull results from :attr:`.csv_lig` or :atr:`.csv_qd`.

        Performs in inplace update of **df** if **inplace** = ``True``, thus returing ``None``.

        Parameters
        ----------
        df : |pd.DataFrame|_
            A dataframe of new (potential) database entries.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"QD"``.

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
        if database == 'ligand':
            path = self.csv_lig
            open_csv = self.open_csv_lig
        elif database == 'QD':
            path = self.csv_qd
            open_csv = self.open_csv_qd

        # Update the *hdf5 index* column in **df**
        with open_csv(path, write=False) as db:
            df.update(db['df'], overwrite=True)
            df['hdf5 index'] = df['hdf5 index'].astype(int, copy=False)

        # **df** has been updated and **get_mol** = *False*
        if get_mol:
            ret = self._get_csv_mol(df, database, inplace)
        else:
            ret = None

        # Return a new series if **inplace** = *False*; return *None* otherwise
        return ret

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
            The type of database; accepted values are ``"ligand"`` and ``"QD"``.

        inplace : bool
            If ``True`` perform an inplace update of the ``("mol", "")`` column in **df**.
            Otherwise return a new series of PLAMS molecules.

        Returns
        -------
        |pd.Series|_ [|plams.Molecule|_]
            Optional: A Series of PLAMS molecules if **inplace** is ``False``.

        """
        # Sort and find all valid HDF5 indices
        df.sort_values(by=['hdf5 index'], inplace=True)
        df_slice = df['opt'] == True  # noqa
        idx = df['hdf5 index'][df_slice].values

        # If no HDF5 indices are availble in **df** then abort the function
        if not df_slice.any():
            if inplace:
                return None
            return pd.Series(None, name=('mol', ''), dtype=object)

        # Update **df** with preexisting molecules from **self**, returning *None*
        if inplace:
            mol_list = self.from_hdf5(idx, database=database)
            for i, rdmol in zip(df_slice.index, mol_list):
                df.loc[i, ('mol', '')].from_rdmol(rdmol)
            ret = None

        # Create and return a new series of PLAMS molecules
        else:
            mol_list = self.from_hdf5(idx, database=database, rdmol=False)
            ret = pd.Series(mol_list, index=df[df_slice].index, name=('mol', ''))

        return ret

    def from_hdf5(self, index: Sequence[int],
                  database: str = 'ligand',
                  rdmol: bool = True,
                  close: bool = True) -> List[Union[Molecule, Mol]]:
        """Import structures from the hdf5 database as RDKit or PLAMS molecules.

        Parameters
        ----------
        index : |list|_ [|int|_]
            The indices of the to be retrieved structures.

        database : str
            The type of database; accepted values are ``"ligand"`` and ``"QD"``.

        rdmol : bool
            If ``True``, return an RDKit molecule instead of a PLAMS molecule.

        close : bool
            If the database component (:attr:`.hdf5`) should be closed afterwards.

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
        with h5py.File(self.hdf5, 'r') as f:
            pdb_array = f[database][index]

        # Return a list of RDKit or PLAMS molecules
        return [from_pdb_array(mol, rdmol=rdmol) for mol in pdb_array]

    def hdf5_availability(self, timeout: float = 5.0,
                          max_attempts: Optional[int] = None) -> None:
        """Check if a .hdf5 file is opened by another process; return once it is not.

        If two processes attempt to simultaneously open a single hdf5 file then
        h5py will raise an :class:`OSError`.
        The purpose of this method is ensure that a .hdf5 file is actually closed,
        thus allowing the :meth:`.from_hdf5` method to safely access **filename** without
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
        warning = "OSWarning: '{}' is currently unavailable; repeating attempt in {:.0f} seconds"
        i = max_attempts or np.inf

        while i:
            try:
                with h5py.File(self.hdf5, 'r+', libver='latest') as _:
                    return None  # the .hdf5 file can safely be opened
            except OSError as ex:  # the .hdf5 file cannot be safely opened yet
                print((warning).format(self.hdf5, timeout))
                error = ex
                sleep(timeout)
            i -= 1
        raise error
