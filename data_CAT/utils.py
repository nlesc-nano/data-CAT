"""Various utilities."""

import time
import pkg_resources as pkg
from os.path import join
from typing import (List, Iterable, Optional, Settings)

import yaml

from rdkit.Chem import Mol
from scm.plams import (add_to_class, Molecule, Atom)

__all__: List[str] = ['get_time', 'get_template']


def get_time() -> str:
    """Return the current time as a :class:`str`."""
    return '[{}] '.format(time.strftime('%H:%M:%S'))


def get_template(filename: str,
                 from_cat_data: bool = True) -> Settings:
    """Grab a yaml template and return its content as :class:`.Settings` instance.

    Paramaters
    ----------
    filename : |str|_
        The path+filename of a .yaml file.

    from_cat_data : |bool|_
        Whether or not **filename** is in the `"/data_CAT/data/templates"` directory.

    Returns
    -------
    |plams.Settings|_
        A :class:`.Settings` instance constructed from **filename**.

    """
    if from_cat_data:
        path = join('data/templates', filename)
        xs = pkg.resource_string('data_CAT', path)
        return Settings(yaml.load(xs.decode(), Loader=yaml.FullLoader))
    else:
        with open(filename, 'r') as file:
            return Settings(yaml.load(file, Loader=yaml.FullLoader))


@add_to_class(Molecule)
def from_rdmol(self, rdmol: Mol,
               atom_subset: Optional[Iterable[Atom]] = None) -> None:
    """ Update the atomic coordinates of this instance with coordinates from an RDKit molecule.

    Performs an inplace update of all coordinates (:attr:`.Atom.coords`) in
    this instance (:attr:`.Molecule.atoms`).

    Paramaters
    ----------
    rdmol : |rdkit.Chem.Mol|_
        An RDKit molecule.

    atom_subset : |Iterable|_ [|plams.Atom|_]
        Optional: A subset of atoms in this :class:`Molecule` instance.

    Note
    ----
    Atoms provided in **atom_subset** should be sorted in the same manner as
    the atoms in **rdmol**.

    """
    at_subset = atom_subset or self.atoms
    conf = rdmol.GetConformer()
    for at1, at2 in zip(at_subset, rdmol.GetAtoms()):
        pos = conf.GetAtomPosition(at2.GetIdx())
        at1.coords = (pos.x, pos.y, pos.z)
