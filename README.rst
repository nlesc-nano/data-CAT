.. image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://www.python.org


##############
Data-CAT 0.1.0
##############

Data-CAT is a databasing framework for the Compound Attachment Tools package (CAT_).


Installation
============

- Download miniconda for python3: miniconda_ (also you can install the complete anaconda_ version).

- Install according to: installConda_.

- Create a new virtual environment, for python 3.7, using the following commands:

  - ``conda create --name CAT python``

- The virtual environment can be enabled and disabled by, respectively, typing:

  - Enable: ``conda activate CAT``

  - Disable: ``conda deactivate``


Dependencies installation
-------------------------

Using the conda environment the following packages should be installed:

- rdkit_ & h5py_: ``conda install -y --name CAT --channel conda-forge rdkit h5py``


Package installation
--------------------
Finally, install **Data-CAT** using pip:

- **Data-CAT**: ``pip install git+https://github.com/nlesc-nano/Data-CAT@master --upgrade``

Now you are ready to use **Data-CAT**.


.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: https://www.continuum.io/downloads
.. _installConda: https://docs.anaconda.com/anaconda/install/
.. _CAT: https://github.com/nlesc-nano/CAT
.. _rdkit: http://www.rdkit.org
.. _h5py: http://www.h5py.org/
