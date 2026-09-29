"""Atom-to-bead space partitioning (Voronoi-style).

Given CG bead headliner positions (one heavy atom per bead), assigns every
other heavy atom -- and then every hydrogen -- to its nearest bead. Two
variants exist: voronoi_atoms_new() (the default, used for molecules with
few fused rings) and voronoi_atoms_old() (used as a fallback for heavily
fused-ring molecules, or when a mapping forced by the caller requires it).
"""

from ..common import *

logger = logging.getLogger(__name__)


def _heavy_neighbor_map(molecule, num_heavy, num_atoms):
    """Precompute, once per molecule, which heavy-atom local index (0..num_heavy-1)
    each other atom (e.g. hydrogens) is bonded to. voronoi_atoms_new()/voronoi_atoms_old()
    use this to lump those atoms into the same bead as their bonded heavy atom -- this
    mapping is independent of the trial bead placement, so it's wasteful to rebuild it
    (by re-scanning and re-parsing every molecule bond) on every call, as the original
    code did. Mirrors the original bond-string-scan order exactly (last matching bond
    for a given atom wins) so caching it does not change the result."""
    bonds = []
    for b in range(molecule.GetNumBonds()):
        abond = molecule.GetBondWithIdx(b)
        at1 = abond.GetBeginAtomIdx()
        at2 = abond.GetEndAtomIdx()
        if f"{at1}-{at2}" not in bonds and f"{at2}-{at1}" not in bonds:
            bonds.append(f"{at1}-{at2}")

    heavy_keys = range(num_heavy)
    neighbor_of_atom = {}
    for at in range(num_atoms):
        if at in heavy_keys:
            continue
        hbead = None
        for b in bonds:
            bond = b.split('-')
            if str(at) in bond:
                at1 = int(bond[0])
                at2 = int(bond[-1])
                if at == at1 and at2 < num_heavy:
                    hbead = at2
                if at == at2 and at1 < num_heavy:
                    hbead = at1
                if hbead is not None:
                    neighbor_of_atom[at] = hbead
    return neighbor_of_atom


def _bead_atom_tables(cgbead_coords, heavyatom_coords):
    """Per bead: its distance to every heavy atom, and whether they coincide."""
    dist = [[float(np.linalg.norm(c - a)) for a in heavyatom_coords] for c in cgbead_coords]
    same = [[bool((a == c).all()) for a in heavyatom_coords] for c in cgbead_coords]
    return dist, same


def _closest_rank(dist_row, same_row):
    """Rank (0 = closest) of every heavy atom not coinciding with the bead,
    ties kept in atom-index order."""
    distances = {j: d for j, d in enumerate(dist_row) if not same_row[j]}
    return {atom: rank for rank, atom in enumerate(sorted(distances, key=distances.get))}


class HeavyAtomGeometry:
    """Pairwise heavy-atom distances and coincidences, computed once per molecule
    so the bead search doesn't recompute them for every trial combination.

    During the search every bead headliner *is* a heavy atom, so each distance
    the partitioning needs is np.linalg.norm(heavy[h] - heavy[j]) for some pair.
    Computing each pair once with that same call gives bit-identical values. (A
    hand-written scalar sqrt does not: it differs from numpy's in the last bit
    for ~9% of pairs, enough to flip the distance ties broken by sort order.)
    """

    def __init__(self, heavyatom_coords):
        self.dist, self.same = _bead_atom_tables(heavyatom_coords, heavyatom_coords)
        self._closest_rank = {}

    def closest_rank(self, h):
        if h not in self._closest_rank:
            self._closest_rank[h] = _closest_rank(self.dist[h], self.same[h])
        return self._closest_rank[h]


def _bead_rows(cgbead_coords, heavyatom_coords, geometry, bead_heavy_idx):
    """Distance/coincidence rows for the given beads: looked up in the
    precomputed geometry when the beads are known heavy atoms, else computed."""
    if geometry is not None and bead_heavy_idx is not None:
        return (
            [geometry.dist[h] for h in bead_heavy_idx],
            [geometry.same[h] for h in bead_heavy_idx],
        )
    return _bead_atom_tables(cgbead_coords, heavyatom_coords)


def _absorb_lonely_beads(partitioning, closest_atoms, bead_dist, num_heavy):
    """If one bead has only one heavy atom, include one more: the closest atom
    (not itself a bead headliner) from a bead that has more than two."""
    # Kept in sync with partitioning at its single mutation point below, so
    # every lookup equals counting partitioning.values() at that moment.
    bead_size = Counter(partitioning.values())
    for i in list(partitioning.values()):
        if bead_size[i] == 1:
            # Find bead
            lonely_bead = i
            # Voronoi to find closest atom
            closest_bead = -1
            closest_bead_dist = 10000.0
            for j in range(num_heavy):
                if partitioning[j] != lonely_bead:
                    dist_bead_at = bead_dist[lonely_bead][j]
                    # Only consider if it's closer, not a CG bead itself, and
                    # the CG bead it belongs to has more than one other atom.
                    if (
                        dist_bead_at < closest_bead_dist
                        and j != closest_atoms[partitioning[j]]
                        and bead_size[partitioning[j]] > 2
                    ):
                        closest_bead = j
                        closest_bead_dist = dist_bead_at
            if closest_bead == -1:
                logger.warning("Error. Can't find an atom close to atom $s" % lonely_bead)
                exit(1)
            bead_size[partitioning[closest_bead]] -= 1
            bead_size[lonely_bead] += 1
            partitioning[closest_bead] = lonely_bead


def _hydrogen_aware_cog(partitioning, allatom_coords, heavy_neighbor_of):
    """Center of geometry of each bead, including the hydrogens bonded to its heavy atoms."""
    aa_partitioning = partitioning.copy()
    for at, heavy_local_idx in heavy_neighbor_of.items():
        aa_partitioning[at] = partitioning[heavy_local_idx]

    bead_coord={}
    for atom in range(len(allatom_coords)):
        bead=aa_partitioning[atom]
        if bead not in bead_coord.keys():
            bead_coord[bead]=[]
        bead_coord[bead].append(allatom_coords[atom])

    bead_cog=[]
    for bead, coords in sorted(bead_coord.items()):
        cog = np.mean(coords,axis=0)
        bead_cog.append(cog)
    return bead_cog


def voronoi_atoms_new(
    cgbead_coords, heavyatom_coords, allatom_coords, molecule, heavy_neighbor_of=None,
    geometry=None, bead_heavy_idx=None, with_cog=True,
):
    """
    Partition all atoms between CG beads, based on headliners coordinates and distances between other atoms coordinates.
    Headliners are atoms with cgbead_coords coordinates.

    geometry/bead_heavy_idx: optional HeavyAtomGeometry, and each bead's index in
    heavyatom_coords, when every bead sits on a heavy atom (as in the bead search).
    with_cog: set False to skip computing bead centers when only the partitioning is needed.
    """
    logger.debug("Entering voronoi_atoms()")
    bead_dist, bead_same = _bead_rows(cgbead_coords, heavyatom_coords, geometry, bead_heavy_idx)
    partitioning = {}

    #Populate partitioning with atoms and atom headliners of beads
    for j in range(len(heavyatom_coords)):
        partitioning[j] = None
        for b in range(len(cgbead_coords)):
            if bead_same[b][j]:
                partitioning[j] = b

    # Find closest atoms to atom headliners of beads
    if len(cgbead_coords) > 1:
        # For every bead, the rank (0 = closest) of each other heavy atom
        closest_atoms = {}
        for i in range(len(cgbead_coords)):
            if geometry is not None and bead_heavy_idx is not None:
                closest_atoms[i] = geometry.closest_rank(bead_heavy_idx[i])
            else:
                closest_atoms[i] = _closest_rank(bead_dist[i], bead_same[i])

        # Populate partitioning with closest atoms
        for atom, bead in partitioning.items():

            if bead is None:
                closest_index = float('inf')  # Initialize with infinity
                closest_bead = None

                for current_bead, atom_rank in closest_atoms.items():
                    if atom in atom_rank:
                        index = atom_rank[atom]
                        if index < closest_index:
                            closest_index = index
                            closest_bead = current_bead

                if closest_bead is not None:
                    partitioning[atom] = closest_bead

        _absorb_lonely_beads(partitioning, closest_atoms, bead_dist, len(heavyatom_coords))
    else:
        for j in range(len(heavyatom_coords)):
            partitioning[j] = 0

    if not with_cog:
        return partitioning, None
    if heavy_neighbor_of is None:
        heavy_neighbor_of = _heavy_neighbor_map(molecule, len(heavyatom_coords), len(allatom_coords))
    return partitioning, _hydrogen_aware_cog(partitioning, allatom_coords, heavy_neighbor_of)


def voronoi_atoms_old(
    cgbead_coords, heavyatom_coords, allatom_coords, molecule, heavy_neighbor_of=None,
    geometry=None, bead_heavy_idx=None, with_cog=True,
):
    """Partition all atoms between CG beads

    geometry/bead_heavy_idx/with_cog: see voronoi_atoms_new().
    """
    logger.debug("Entering voronoi_atoms()")
    bead_dist, _ = _bead_rows(cgbead_coords, heavyatom_coords, geometry, bead_heavy_idx)
    partitioning = {}
    for j in range(len(heavyatom_coords)):
        if j not in partitioning.keys():
            # Voronoi to check whether atom is closest to bead
            bead_at = -1
            dist_bead_at = 1000
            for k in range(len(cgbead_coords)):
                distk = bead_dist[k][j]
                if distk < dist_bead_at:
                    dist_bead_at = distk
                    bead_at = k
            partitioning[j] = bead_at
    if len(cgbead_coords) > 1:
        # Book-keeping of closest atoms to every bead
        closest_atoms = {}
        for i in range(len(cgbead_coords)):
            closest_atom = -1
            closest_dist = 10000.0
            for j in range(len(heavyatom_coords)):
                dist_bead_at = bead_dist[i][j]
                if dist_bead_at < closest_dist:
                    closest_dist = dist_bead_at
                    closest_atom = j
            if closest_atom == -1:
                logger.warning("Error. Can't find closest atom to bead %s" % i)
                exit(1)
            closest_atoms[i] = closest_atom
        _absorb_lonely_beads(partitioning, closest_atoms, bead_dist, len(heavyatom_coords))

    if not with_cog:
        return partitioning, None
    if heavy_neighbor_of is None:
        heavy_neighbor_of = _heavy_neighbor_map(molecule, len(heavyatom_coords), len(allatom_coords))
    return partitioning, _hydrogen_aware_cog(partitioning, allatom_coords, heavy_neighbor_of)
