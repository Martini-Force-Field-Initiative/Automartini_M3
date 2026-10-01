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

This package used to be a single optimization.py file; it's now split by
responsibility so each piece can be read (and tested) on its own:

- energy             -- the Gaussian bead-placement objective function
- partition          -- Voronoi-style atom-to-bead assignment
- functional_groups  -- Ertl functional-group detection and its mapping checks
- search             -- find_bead_pos(), the combinatorial search entry point

Every name below is re-exported so `auto_martiniM3.optimization.<name>` keeps
working exactly as before the split.
"""

from .energy import (
    read_bead_params,
    gaussian_overlap,
    atoms_in_gaussian,
    penalize_lonely_atoms,
    eval_gaussian_interac,
)
from .partition import fused_ring_groups, voronoi_atoms_new, voronoi_atoms_old
from .functional_groups import (
    merge,
    identify_functional_groups,
    functional_groups_ok,
    max2arperbead,
)
from .search import check_beads, find_bead_pos, all_atoms_in_beads_connected, worker_count

__all__ = [
    "read_bead_params",
    "gaussian_overlap",
    "atoms_in_gaussian",
    "penalize_lonely_atoms",
    "eval_gaussian_interac",
    "fused_ring_groups",
    "voronoi_atoms_new",
    "voronoi_atoms_old",
    "merge",
    "identify_functional_groups",
    "functional_groups_ok",
    "max2arperbead",
    "check_beads",
    "find_bead_pos",
    "all_atoms_in_beads_connected",
    "worker_count",
]
