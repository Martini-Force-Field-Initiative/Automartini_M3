"""
Created on March 13, 2019 by Andrew Abi-Mansour
Updated to Martini 3 force field on January 31, 2025 by Magdalena Szczuka

This is the::
    _   _   _ _____ ___     __  __    _    ____ _____ ___ _   _ ___   __  __ _____
   / \ | | | |_   _/ _ \   |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|  |  \/  |___ /
  / _ \| | | | | || | | |  | |\/| | / _ \ | |_) || |  | ||  \| || |   | |\/| | |_ \
 / ___ \ |_| | | || |_| |  | |  | |/ ___ \|  _ < | |  | || |\  || |   | |  | |___) |
/_/  _\_\___/  |_| \___/   |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|  |_|  |_|____/


A tool for automatic MARTINI 3 force field mapping and parametrization of small organic molecules

Developers::
        Magdalena Szczuka (magdalena.szczuka at univ-tlse3.fr)
        Tristan BEREAU (bereau at mpip-mainz.mpg.de)
        Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
        Andrew Abi-Mansour (andrew.gaam at gmail.com)

AUTO_MARTINI M3 is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.

--

This package used to be a single topology.py file; it's now split by
responsibility so each piece can be read (and tested) on its own:

- molecule    -- build an RDKit molecule from SMILES/SDF, extract atoms/features
- beadtypes   -- Martini 3 delta_f database and CG bead-type assignment
- alogps      -- fragment logP lookup (local database + ALOGPS web service)
- topout      -- .itp topology text generation

Every name below is re-exported so `auto_martiniM3.topology.<name>` keeps
working exactly as before the split.
"""

from auto_martiniM3._version import __version__

from .molecule import (
    fdefName,
    factory,
    gen_molecule_smi,
    gen_molecule_sdf,
    heavy_atoms_first,
    get_charge,
    get_hbond_a,
    get_hbond_d,
    get_atoms,
    get_ring_atoms,
    is_aromatic,
    get_heavy_atom_coords,
    extract_features,
)
from .beadtypes import (
    read_delta_f_types,
    letter_occurrences,
    cyclic_smi_conversion,
    find_closest_key,
    rearrange_until_match,
    read_params,
    substruct2smi,
    get_mass,
    get_standard_mass,
    mad,
    count_letters,
    find_closest_logPvalue,
    determine_bead_type,
)
from .alogps import smi2alogps, convert_log_k
from .topout import (
    print_header,
    print_atoms,
    print_bonds,
    print_angles,
    print_dihedrals,
    print_virtualsites,
    topout,
    topout_noVS,
    topout_vs,
    bartender_input,
)

__all__ = [
    "__version__",
    "fdefName",
    "factory",
    "gen_molecule_smi",
    "gen_molecule_sdf",
    "heavy_atoms_first",
    "get_charge",
    "get_hbond_a",
    "get_hbond_d",
    "get_atoms",
    "get_ring_atoms",
    "is_aromatic",
    "get_heavy_atom_coords",
    "extract_features",
    "read_delta_f_types",
    "letter_occurrences",
    "cyclic_smi_conversion",
    "find_closest_key",
    "rearrange_until_match",
    "read_params",
    "substruct2smi",
    "get_mass",
    "get_standard_mass",
    "mad",
    "count_letters",
    "find_closest_logPvalue",
    "determine_bead_type",
    "smi2alogps",
    "convert_log_k",
    "print_header",
    "print_atoms",
    "print_bonds",
    "print_angles",
    "print_dihedrals",
    "print_virtualsites",
    "topout",
    "topout_noVS",
    "topout_vs",
    "bartender_input",
]
