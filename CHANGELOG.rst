###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


0.7.0
*****
* Add a new ``PDBContainer`` method for creating rdkit molecules.
* Expose the ``index_to_pandas`` function.
* Various fixes.


0.5.1
*****
* Fixed an issue where not all properties were copied from the .csv to the .hdf5 file.


0.5.0
*****
* Removed the .yaml job recipe storage component of the `Database` class.
* Added a function for validating property Groups.
* Cleaned up the `Database` class.
* Ensure that all content written to the .csv files is also written to the .hdf5 file.
* Pass the `"logger"` group directly to the `hdf5_log` functions.
* Added `prop_to_dataframe()`, a function for converting property Datasets into DataFrames.
* Added set-based methods to `PDBContainer`: `.union()`, `.intersection()`,
  `.difference()` & `.symmetric_difference()`.
* Added `PDBContainer.concatenate()`.


0.4.0
*****
* Updated the hdf5 logging system.
* Moved all dtype-related functions and data to `dataCAT.dtype`.
* Cleaned up the global namespace.
* Added the `PDBContainer.index` attribute.
* Create the `index` Dataset when calling `PDBContainer.create_hdf5_group()`.
  Aforementioned Dataset is used as dimensional scale.
* Provide users more control over the keyword arguments used for creating datasets.
* Fixed an issue where`PDBContainer.to_molecules()` would set values as numpy scalars
  rather than using its builtin counterparts.
* Added an API for creating and managing datasets for containing quantum-mechanical
  properties in hdf5 format.
* Cleaned up the backwards-compatibility checks in `dataCAT.create_database`.


0.3.1
*****
* Fixed bug with the `PDBContainer` hashing system.
* Moved part of the hdf5 database creation process to `PDBContainer.create_hdf5_group()`.
* Added `PDBContainer.keys()` and `PDBContainer.values()`.
* Moved a number of test files to `dataCAT.testing_utils` and `dataCAT.data`.
* Ignore flake8 `N806`.
* Added a test for building wheels.
* Added `h5py` to `install_requires`.
* Check for the presence of `rdkit` when running `setup.py`.
* Added `PDBContainer.validate_hdf5()`.
* Execute `.validate_hdf5()` automatically when `.to_hdf5()` or `.from_hdf5()` crashes.
* Moved `hdf5_availability()` from `Database` to `dataCAT.functions`.
* Cleaned up the global namespace.
* Added `libver="latest"` to `Database.hdf5`.


0.3.0
*****
* Overhauled the .pdb storage system.
* Introduced the `PDBContainer` class.
* Renamed `dataCAT.database_functions` to `dataCAT.functions`.


0.2.2
*****
* Updated the documentation (see https://github.com/nlesc-nano/CAT/pull/123).


0.2.1
*****
* Store the ``__version__`` of CAT, Nano-CAT and Data-CAT in the hdf5 file.
* Store the content of the .csv file also in the .hdf5 file.
* Reach a test coverage of >= 80%.


0.2.0
*****
* Moved from travis to GitHub Actions.
* Enabled tests for Python 3.8.
* Added tests using ``flake8`` and ``pydocstyle``.
* Removed the unused `requirements.txt` file.
* Cleaned up `setup.py`.
* Cleaned up the context managers; removed the `MetaManager` class in favor of `functools.partial`.
* Cleaned up the ``DFCollection`` class; renamed it to ``DFProxy``.


0.1.5
*****
* Decapitalized all references to ``"QD"``.
* Import assertions from AssertionLib_ rather than CAT_.


0.1.4
*****
* Updated the handling of assertions, see ``CAT.assertions.assertion_manager``.


0.1.3
*****
* Lowered Python version requirement from >=3.7 to >=3.6.
* Changed the ``dataCAT.Metamanager()`` class from a dataclass
  into a subclass of ``collections.abc.Container()``


0.1.2
*****
* Updated many ``__str__`` and ``__repr__`` methods.
* Added the ``Database.__eq__`` method.
* Moved context managers to ``dataCAT.context_managers``
* Moved (and renamed) the ``DF()`` class to ``dataCAT.df_collection.DFCollection()``.
* Added more tests.


0.1.1
*****
* Introduced a proper logger (see https://github.com/nlesc-nano/CAT/issues/46 and
  https://github.com/nlesc-nano/CAT/pull/47).


[Unreleased]
************
* Empty Python project directory structure.


.. _AssertionLib: https://github.com/nlesc-nano/AssertionLib
.. _CAT: https://github.com/nlesc-nano/CAT
