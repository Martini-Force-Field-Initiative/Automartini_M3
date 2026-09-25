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


def voronoi_atoms_new(cgbead_coords, heavyatom_coords, allatom_coords, molecule, heavy_neighbor_of=None):
    """
    Partition all atoms between CG beads, based on headliners coordinates and distances between other atoms coordinates.
    Headliners are atoms with cgbead_coords coordinates.
    """
    logger.debug("Entering voronoi_atoms()")
    partitioning = {}

    #Populate partitioning with atoms and atom headliners of beads
    for j in range(len(heavyatom_coords)):
        partitioning[j] = None
        for b in range(len(cgbead_coords)):
            if(heavyatom_coords[j]==cgbead_coords[b]).all():
                partitioning[j] = b

    # Find closest atoms to atom headliners of beads
    if len(cgbead_coords) > 1:
        closest_atoms = {}  # Book-keeping of closest atoms to every bead
        for i in range(len(cgbead_coords)):
            distances = {}
            for j in range(len(heavyatom_coords)):
                if (cgbead_coords[i] != heavyatom_coords[j]).any():
                    dist_bead_at = np.linalg.norm(cgbead_coords[i] - heavyatom_coords[j])
                    distances[j] = dist_bead_at  # Atom index as key, distance as value

            # Sort distances by value and keep the closest atoms
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            closest_atoms[i] = sorted_distances  # Dictionary of atoms and their distances for each bead

        # Populate partitioning with closest atoms
        for atom, bead in partitioning.items():

            if bead is None:
                closest_index = float('inf')  # Initialize with infinity
                closest_bead = None

                for current_bead, atoms_dict in closest_atoms.items():
                    if atom in atoms_dict:
                        index = list(atoms_dict.keys()).index(atom)  # Find the index of the atom in the sorted keys
                        if index < closest_index:
                            closest_index = index
                            closest_bead = current_bead

                if closest_bead is not None:
                    partitioning[atom] = closest_bead

        # If one bead has only one heavy atom, include one more
        for i in list(partitioning.values()):
            if sum(x == i for x in partitioning.values()) == 1:
                # Find bead
                lonely_bead = i
                # Voronoi to find closest atom
                closest_bead = -1
                closest_bead_dist = 10000.0
                for j in range(len(heavyatom_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - heavyatom_coords[j]
                        )
                        # Only consider if it's closer, not a CG bead itself, and
                        # the CG bead it belongs to has more than one other atom.
                        if (
                            dist_bead_at < closest_bead_dist
                            and j != closest_atoms[partitioning[j]]
                            and sum(x == partitioning[j] for x in partitioning.values()) > 2
                        ):
                            closest_bead = j
                            closest_bead_dist = dist_bead_at
                if closest_bead == -1:
                    logger.warning("Error. Can't find an atom close to atom $s" % lonely_bead)
                    exit(1)
                partitioning[closest_bead] = lonely_bead
    else:
        for j in range(len(heavyatom_coords)):
            partitioning[j] = 0

    if heavy_neighbor_of is None:
        heavy_neighbor_of = _heavy_neighbor_map(molecule, len(heavyatom_coords), len(allatom_coords))

    # create partitioning including hydrogens inside beads
    aa_partitioning = partitioning.copy()
    for at, heavy_local_idx in heavy_neighbor_of.items():
        aa_partitioning[at] = partitioning[heavy_local_idx]

    #compute COG while taking into account hydrogens
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

    return partitioning, bead_cog


def voronoi_atoms_old(cgbead_coords, heavyatom_coords, allatom_coords, molecule, heavy_neighbor_of=None):
    """Partition all atoms between CG beads"""
    logger.debug("Entering voronoi_atoms()")
    partitioning = {}
    for j in range(len(heavyatom_coords)):
        if j not in partitioning.keys():
            # Voronoi to check whether atom is closest to bead
            bead_at = -1
            dist_bead_at = 1000
            for k in range(len(cgbead_coords)):
                distk = np.linalg.norm(cgbead_coords[k] - heavyatom_coords[j])
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
                dist_bead_at = np.linalg.norm(cgbead_coords[i] - heavyatom_coords[j])
                if dist_bead_at < closest_dist:
                    closest_dist = dist_bead_at
                    closest_atom = j
            if closest_atom == -1:
                logger.warning("Error. Can't find closest atom to bead %s" % i)
                exit(1)
            closest_atoms[i] = closest_atom
        # If one bead has only one heavy atom, include one more
        for i in list(partitioning.values()):
            if sum(x == i for x in partitioning.values()) == 1:
                # Find bead
                lonely_bead = i
                # Voronoi to find closest atom
                closest_bead = -1
                closest_bead_dist = 10000.0
                for j in range(len(heavyatom_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - heavyatom_coords[j]
                        )
                        # Only consider if it's closer, not a CG bead itself, and
                        # the CG bead it belongs to has more than one other atom.
                        if (
                            dist_bead_at < closest_bead_dist
                            and j != closest_atoms[partitioning[j]]
                            and sum(x == partitioning[j] for x in partitioning.values()) > 2
                        ):
                            closest_bead = j
                            closest_bead_dist = dist_bead_at
                if closest_bead == -1:
                    logger.warning("Error. Can't find an atom close to atom $s" % lonely_bead)
                    exit(1)
                partitioning[closest_bead] = lonely_bead

    if heavy_neighbor_of is None:
        heavy_neighbor_of = _heavy_neighbor_map(molecule, len(heavyatom_coords), len(allatom_coords))

    # create partitioning including hydrogens inside beads
    aa_partitioning = partitioning.copy()
    for at, heavy_local_idx in heavy_neighbor_of.items():
        aa_partitioning[at] = partitioning[heavy_local_idx]

    #compute COG while taking into account hydrogens
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

    return partitioning, bead_cog
