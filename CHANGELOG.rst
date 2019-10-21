###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


0.1.5
*****
* Decapitalized all references to ``"QD"``.


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

Added
-----

* Empty Python project directory structure.
