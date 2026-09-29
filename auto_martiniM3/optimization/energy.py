"""Gaussian bead-placement objective function.

Scores a candidate set of CG bead positions: an offset cost per bead, a
repulsive penalty for overlapping beads, an attractive term for atoms lumped
into a bead, and a penalty for atoms left out of every bead. find_bead_pos()
minimizes the sum of these over all trial bead combinations.
"""

from ..common import *

logger = logging.getLogger(__name__)


_BEAD_PARAMS = {
    "rvdw": 4.7 / 2.0,  # sigma for non-ring
    "rvdw_aromatic": 4.1 / 2.0,  # sigma for ring
    "rvdw_cross": 0.5 * ((4.7 / 2.0) + (4.3 / 2.0)),
    "offset_bd_weight": 20.0,  # penalty weight for nonring beads
    "offset_bd_aromatic_weight": 5.0,  # penalty weight for ring beads
    "lonely_atom_penalize": 0.28,
    "bd_bd_overlap_coeff": 1.0,
    "at_in_bd_coeff": 0.9,
}


def read_bead_params():
    """Returns bead parameter dictionary
    CG Bead vdw radius (in Angstroem)"""
    return _BEAD_PARAMS


def gaussian_overlap(conformer, bead1, bead2, ringatoms):
    """Returns overlap coefficient between two gaussians
    given distance dist"""
    logger.debug("Entering gaussian_overlap()")
    dist = Chem.rdMolTransforms.GetBondLength(conformer, int(bead1), int(bead2))
    bead_params = read_bead_params()
    sigma = bead_params["rvdw"]
    if bead1 in ringatoms and bead2 in ringatoms:
        sigma = bead_params["rvdw_aromatic"]
    if (
        bead1 in ringatoms
        and bead2 not in ringatoms
        or bead1 not in ringatoms
        and bead2 in ringatoms
    ):
        sigma = bead_params["rvdw_cross"]
    return bead_params["bd_bd_overlap_coeff"] * math.exp(-(dist**2) / 4.0 / sigma**2)


def atoms_in_gaussian(molecule, conformer, bead_id, ringatoms):
    """Returns weighted sum of atoms contained in bead bead_id"""
    logger.debug("Entering atoms_in_gaussian()")
    weight_sum = 0.0
    bead_params = read_bead_params()
    sigma = bead_params["rvdw"]
    lumped_atoms = []
    if bead_id in ringatoms:
        sigma = bead_params["rvdw_aromatic"]
    for i in range(conformer.GetNumAtoms()):
        dist_bd_at = Chem.rdMolTransforms.GetBondLength(conformer, i, int(bead_id))
        if dist_bd_at < sigma:
            lumped_atoms.append(i)
        weight_sum -= molecule.GetAtomWithIdx(i).GetMass() * math.exp(
            -(dist_bd_at**2) / 2 / sigma**2
        )
    return bead_params["at_in_bd_coeff"] * weight_sum, lumped_atoms


def penalize_lonely_atoms(molecule, conformer, lumped_atoms, masses=None):
    """Penalizes configuration if atoms aren't included
    in any CG bead"""
    logger.debug("Entering penalize_lonely_atoms()")
    weight_sum = 0.0
    bead_params = read_bead_params()
    num_atoms = conformer.GetNumAtoms()
    lumped_atoms_set = set(int(a) for a in lumped_atoms)
    for i in range(num_atoms):
        if i not in lumped_atoms_set:
            weight_sum += masses[i] if masses is not None else molecule.GetAtomWithIdx(i).GetMass()
    return bead_params["lonely_atom_penalize"] * weight_sum


class GaussianTermCache:
    """Memoizes, for one molecule/conformer/ring-atom set, the terms of
    eval_gaussian_interac() that only depend on the atoms involved: the overlap
    of a bead pair, the atoms lumped into a bead, and the atomic masses.

    find_bead_pos() scores thousands of trial combinations built from the same
    ~20 heavy atoms, so without this the same few hundred terms get recomputed
    tens of thousands of times. Each cached value comes from the very same
    function call on the very same inputs, so results are unchanged.
    """

    def __init__(self, molecule, conformer, ringatoms):
        self.molecule = molecule
        self.conformer = conformer
        self.ringatoms = ringatoms
        self.masses = [molecule.GetAtomWithIdx(i).GetMass() for i in range(conformer.GetNumAtoms())]
        self._overlap = {}
        self._in_gaussian = {}

    def overlap(self, bead1, bead2):
        key = (int(bead1), int(bead2))
        if key not in self._overlap:
            self._overlap[key] = gaussian_overlap(self.conformer, bead1, bead2, self.ringatoms)
        return self._overlap[key]

    def in_gaussian(self, bead_id):
        key = int(bead_id)
        if key not in self._in_gaussian:
            self._in_gaussian[key] = atoms_in_gaussian(self.molecule, self.conformer, bead_id, self.ringatoms)
        return self._in_gaussian[key]


def eval_gaussian_interac(molecule, conformer, list_beads, ringatoms, cache=None):
    """From collection of CG beads placed on mol, evaluate
    objective function of interacting beads.

    cache: optional GaussianTermCache built for this same molecule, conformer
    and ringatoms, to reuse per-bead and per-pair terms across many calls."""
    logger.debug("Entering eval_gaussian_interac()")

    weight_sum = 0.0
    weight_overlap = 0.0
    weight_at_in_bd = 0.0
    bead_params = read_bead_params()

    # Offset energy for every new CG bead.
    # Distinguish between aromatics and others.
    num_aromatics = 0
    lumped_atoms = []

    list_beads_array = np.asarray(list_beads)
    num_beads = list_beads_array.size
    for i in range(num_beads):
        if list_beads_array[i] in ringatoms:
            num_aromatics += 1
    weight_offset_bd_weights = (
        bead_params["offset_bd_weight"] * (num_beads - num_aromatics)
        + bead_params["offset_bd_aromatic_weight"] * num_aromatics
    )
    weight_sum += weight_offset_bd_weights

    # Repulsive overlap between CG beads
    for i in range(num_beads - 1):
        for j in range(i + 1, num_beads):
            if cache is not None:
                weight_overlap += cache.overlap(list_beads_array[i], list_beads_array[j])
            else:
                weight_overlap += gaussian_overlap(
                    conformer, list_beads_array[i], list_beads_array[j], ringatoms
                )
    weight_sum += weight_overlap

    # Attraction between atoms nearby to CG bead
    lumped_atoms_seen = set()
    for i in range(num_beads):
        if cache is not None:
            weight, lumped = cache.in_gaussian(list_beads_array[i])
        else:
            weight, lumped = atoms_in_gaussian(molecule, conformer, list_beads_array[i], ringatoms)
        weight_at_in_bd += weight
        for a in lumped:
            if a not in lumped_atoms_seen:
                lumped_atoms_seen.add(a)
                lumped_atoms.append(a)
    weight_sum += weight_at_in_bd
    # Penalty for excluding atoms
    masses = cache.masses if cache is not None else None
    weight_lonely_atoms = penalize_lonely_atoms(molecule, conformer, lumped_atoms, masses)
    weight_sum += weight_lonely_atoms
    return weight_sum
