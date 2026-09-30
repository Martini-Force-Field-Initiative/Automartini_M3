"""Combinatorial CG bead-placement search.

find_bead_pos() is the entry point: it exhaustively tries every valid
combination of heavy atoms as bead headliners (check_beads() filters out
combinations that can't possibly be valid; all_atoms_in_beads_connected()
rejects combinations whose resulting bead partition isn't fully connected),
scores each surviving combination with the Gaussian objective function, and
returns every valid combination sorted by score. BeadSearch holds the data
computed once per molecule that all those trials share.
"""

from ..common import *
from .. import topology
from .energy import GaussianTermCache, eval_gaussian_interac
from .partition import (HeavyAtomGeometry, _heavy_neighbor_map, fused_ring_groups, voronoi_atoms_new,
                        voronoi_atoms_old)

logger = logging.getLogger(__name__)


def _bond_lookup_aids(listbonds):
    """Precompute, once per molecule, the lookup structures check_beads()
    otherwise rebuilds from listbonds on every one of its many calls:
    a set for O(1) 'is this pair bonded' checks, a count of how many times
    each atom appears as a bond endpoint, and its (ordered) neighbor list."""
    bonds_set = set()
    bond_endpoint_count = Counter()
    neighbor_of = defaultdict(list)
    for a, b in listbonds:
        bonds_set.add(frozenset((a, b)))
        bond_endpoint_count[a] += 1
        bond_endpoint_count[b] += 1
        neighbor_of[a].append(b)
        neighbor_of[b].append(a)
    return bonds_set, bond_endpoint_count, neighbor_of


def check_beads(
    molecule,
    list_heavyatoms,
    heavyatom_coords,
    trial_comb,
    ring_atoms,
    listbonds,
    bonds_set=None,
    bond_endpoint_count=None,
    neighbor_of=None,
):
    """Check if CG bead positions in trailComb are acceptable"""
    logger.debug("Entering check_beads()")
    if bonds_set is None:
        bonds_set, bond_endpoint_count, neighbor_of = _bond_lookup_aids(listbonds)
    acceptable_trial = ""
    # Check for beads at the same place
    count = Counter(trial_comb)
    all_different = True
    for val in count.values():
        if val != 1:
            all_different = False
            acceptable_trial = False
            logger.debug("Error. Multiple beads on the same atom position for %s", trial_comb)
            break
    if all_different:
        acceptable_trial = True
        # Check for beads linked by chemical bond (except in rings)
        bonds_in_rings = [0] * len(ring_atoms)
        for bi in range(len(trial_comb)):
            for bj in range(bi + 1, len(trial_comb)):
                if frozenset((trial_comb[bi], trial_comb[bj])) in bonds_set:
                    bond_in_ring = False
                    for r in range(len(ring_atoms)):
                        if trial_comb[bi] in ring_atoms[r] and trial_comb[bj] in ring_atoms[r]:
                            bonds_in_rings[r] += 1
                            bond_in_ring = True
                    if not bond_in_ring:
                        acceptable_trial = False
                        logger.debug("Error. No bond in ring for %s", trial_comb)
                        break
        if acceptable_trial:
            # Don't allow bonds between atoms of the same ring.
            for bir in range(len(bonds_in_rings)):
                if bonds_in_rings[bir] > 0:
                    logger.debug("Error. Bonds between atoms of the same ring for %s", trial_comb)
                    acceptable_trial = False
        if acceptable_trial:
            # Check for two terminal beads linked by only one atom
            for bi in range(len(trial_comb)):
                for bj in range(bi + 1, len(trial_comb)):
                    if (
                        bond_endpoint_count[trial_comb[bi]] == 1
                        and bond_endpoint_count[trial_comb[bj]] == 1
                    ):
                        # Both beads are on terminal atoms. Block contribution
                        # if the two terminal atoms are linked to the same atom.
                        partneri = neighbor_of[trial_comb[bi]][-1] if neighbor_of[trial_comb[bi]] else ""
                        partnerj = neighbor_of[trial_comb[bj]][-1] if neighbor_of[trial_comb[bj]] else ""
                        if partneri == partnerj:
                            acceptable_trial = False
                            logger.debug(
                                "Error. Two terminal beads linked to the same atom for %s",
                                trial_comb,
                            )
    return acceptable_trial


def _valid_bead_combinations(num_atoms, num_beads, adjacency, terminal_partner):
    """Yield, as tuples of positions in 0..num_atoms-1, exactly the num_beads-sized
    combinations that check_beads() accepts, in itertools.combinations order.

    check_beads() rejects a combination as soon as any two of its atoms are
    bonded (whether or not they share a ring, one branch or the other rejects
    it), or two of its terminal atoms hang off the same neighbor. Both rules are
    pairwise, so a depth-first search that picks positions in increasing order
    can prune a partial combination the moment it breaks one -- instead of
    building every combination and rejecting ~97% of them afterwards.
    """
    combo = []
    blocked = [0] * num_atoms
    partners_in_use = Counter()

    def extend(start):
        if len(combo) == num_beads:
            yield tuple(combo)
            return
        last_start = num_atoms - (num_beads - len(combo))
        for p in range(start, last_start + 1):
            if blocked[p]:
                continue
            partner = terminal_partner[p]
            if partner is not None and partners_in_use[partner]:
                continue
            combo.append(p)
            for q in adjacency[p]:
                blocked[q] += 1
            if partner is not None:
                partners_in_use[partner] += 1
            yield from extend(p + 1)
            combo.pop()
            for q in adjacency[p]:
                blocked[q] -= 1
            if partner is not None:
                partners_in_use[partner] -= 1

    yield from extend(0)


class BeadSearch:
    """Per-molecule data of the bead search, computed once.

    find_bead_pos() scores every valid combination of bead positions -- up to
    about a million for a 30-heavy-atom molecule -- and they all share the same
    bond graph, heavy-atom indexing, geometry, energy terms and fused-ring
    templates. This object holds those, so each trial only does its own work.
    """

    def __init__(self, molecule, conformer, list_heavy_atoms, heavyatom_coords, allatom_coords, ringatoms_flat,
                 force_map):
        self.molecule = molecule
        self.conformer = conformer
        self.list_heavy_atoms = list_heavy_atoms
        self.heavyatom_coords = heavyatom_coords
        self.allatom_coords = allatom_coords
        self.ringatoms_flat = ringatoms_flat
        self.force_map = force_map

        # Bonds between heavy atoms, as [atom, atom] pairs in heavy-atom order.
        self.list_bonds = []
        for i in range(len(list_heavy_atoms)):
            for j in range(i + 1, len(list_heavy_atoms)):
                if molecule.GetBondBetweenAtoms(int(list_heavy_atoms[i]), int(list_heavy_atoms[j])) is not None:
                    self.list_bonds.append([list_heavy_atoms[i], list_heavy_atoms[j]])
        _, bond_endpoint_count, neighbor_of = _bond_lookup_aids(self.list_bonds)
        self.heavy_index = {atom: i for i, atom in enumerate(list_heavy_atoms)}
        _, self.num_arom = topology.is_aromatic(molecule)
        self.heavy_neighbor_of = _heavy_neighbor_map(molecule, len(list_heavy_atoms), len(allatom_coords))

        # Bond graph by position in list_heavy_atoms, and the atom each terminal
        # heavy atom hangs off, for _valid_bead_combinations().
        self.adjacency = [[] for _ in list_heavy_atoms]
        for a, b in self.list_bonds:
            self.adjacency[self.heavy_index[a]].append(self.heavy_index[b])
            self.adjacency[self.heavy_index[b]].append(self.heavy_index[a])
        self.terminal_partner = [
            int(neighbor_of[atom][-1]) if bond_endpoint_count[atom] == 1 else None
            for atom in list_heavy_atoms
        ]

        # Indexing a numpy array keeps trial_comb elements the same type (np.int64)
        # as the original np.array(list(itertools.combinations(...))) rows.
        self.heavy_atoms_array = np.array(list_heavy_atoms)
        self.energy_cache = GaussianTermCache(molecule, conformer, ringatoms_flat)
        self.geometry = HeavyAtomGeometry(heavyatom_coords)
        self.ring_groups = fused_ring_groups(molecule)
        # Atom positions as the Python floats GetAtomPosition() returns.
        self.atom_xyz = [tuple(conformer.GetAtomPosition(i)[m] for m in range(3))
                         for i in range(conformer.GetNumAtoms())]

    def combinations(self, num_beads):
        """Positions (in list_heavy_atoms) of every num_beads-bead combination
        check_beads() accepts, in itertools.combinations order."""
        return _valid_bead_combinations(len(self.list_heavy_atoms), num_beads, self.adjacency,
                                        self.terminal_partner)

    def connected_combinations(self, combinations):
        """Score each combination of positions and yield, in the same order,
        (trial_comb, energy) for those whose beads would hold connected atoms."""
        for positions in combinations:
            trial_comb = list(self.heavy_atoms_array[list(positions)])
            trial_ene = eval_gaussian_interac(self.molecule, self.conformer, trial_comb, self.ringatoms_flat,
                                              cache=self.energy_cache)
            logger.info("; %s %s", trial_comb, trial_ene)
            if all_atoms_in_beads_connected(
                trial_comb, self.heavyatom_coords, self.list_heavy_atoms, self.list_bonds, self.molecule,
                self.allatom_coords, self.force_map, heavy_index=self.heavy_index, num_arom=self.num_arom,
                heavy_neighbor_of=self.heavy_neighbor_of, geometry=self.geometry, ring_groups=self.ring_groups,
            ):
                yield trial_comb, trial_ene

    def bead_positions(self, trial_comb):
        """Coordinates of the beads of trial_comb, in sorted atom order."""
        return [list(self.atom_xyz[int(atom)]) for atom in sorted(trial_comb)]


def find_bead_pos(
    molecule, conformer, list_heavy_atoms, heavyatom_coords, allatom_coords, ring_atoms, ringatoms_flat, force_map
):
    """Try out all possible combinations of CG beads up to threshold number of beads per atom. Find
    arrangement with best energy score. Return all possible arrangements sorted by energy score."""

    logger.debug("Entering find_bead_pos()")

    # Check number of heavy atoms
    if len(list_heavy_atoms) == 0:
        print("Error. No heavy atom found.")
        exit(1)

    if len(list_heavy_atoms) == 1:
        # Put one CG bead on the one heavy atom.
        best_trial_comb = np.array(list(itertools.combinations(range(len(list_heavy_atoms)), 1)))
        avg_pos = [[conformer.GetAtomPosition(best_trial_comb[0])[j] for j in range(3)]]
        return best_trial_comb, avg_pos

    if len(list_heavy_atoms) > 50:
        print("Error. Exhaustive enumeration can't handle large molecules.")
        exit(1)

    search = BeadSearch(molecule, conformer, list_heavy_atoms, heavyatom_coords, allatom_coords, ringatoms_flat,
                        force_map)

    # In Martini 3 a bead covers 2 to 4 heavy atoms.
    min_beads = int(len(list_heavy_atoms) / 4.0)
    max_beads = int(len(list_heavy_atoms) / 2.0)

    candidates = []  # [trial_comb, bead positions, energy] of every connected combination
    best_trial_comb, ene_best_trial = [], 1e6
    last_best_trial_comb = []
    for num_beads in range(min_beads, max_beads + 1):
        # With fewer than 4 heavy atoms min_beads is 0 and that level runs as 1 bead,
        # so the 1-bead level is scored twice and its candidates are listed twice.
        if num_beads == 0:
            num_beads = 1
        for trial_comb, trial_ene in search.connected_combinations(search.combinations(num_beads)):
            if trial_ene < ene_best_trial:
                ene_best_trial = trial_ene
                best_trial_comb = sorted(trial_comb)
            candidates.append([trial_comb, search.bead_positions(trial_comb), trial_ene])

        # Stop adding beads once one more bead no longer changes the best combination.
        if last_best_trial_comb == best_trial_comb:
            break
        last_best_trial_comb = best_trial_comb

    sorted_combs = np.array(sorted(candidates, key=itemgetter(2)), dtype="object")
    return sorted_combs[:, 0], sorted_combs[:, 1]


def all_atoms_in_beads_connected(
    trial_comb, heavyatom_coords, list_heavyatoms, bondlist, mol, allatom_coords, force_map,
    heavy_index=None, num_arom=None, heavy_neighbor_of=None, geometry=None, ring_groups=None,
):
    """Make sure all atoms within one CG bead are connected to at least
    one other atom in that bead"""
    logger.debug("Entering all_atoms_in_beads_connected()")
    if heavy_index is None:
        heavy_index = {atom: i for i, atom in enumerate(list_heavyatoms)}
    # Bead coordinates are given by heavy atoms themselves
    bead_heavy_idx = [heavy_index[atom] for atom in trial_comb]
    cgbead_coords = [heavyatom_coords[h] for h in bead_heavy_idx]

    if num_arom is None:
        _, num_arom = topology.is_aromatic(mol)

    # Only the partitioning is used below, so bead centers can be skipped -- as
    # long as every non-heavy atom has a bonded heavy atom. Otherwise computing
    # them raises KeyError, and that must still happen here: e.g. a SMILES with
    # explicit [H] atoms interleaves hydrogens among the heavy-atom indices.
    with_cog = heavy_neighbor_of is None or (
        len(heavy_neighbor_of) != len(allatom_coords) - len(heavyatom_coords)
    )

    # Molecules with 0-1 fused rings use the newer (faster-converging)
    # partitioning approach; heavily fused-ring molecules, or a mapping the
    # caller is forcing through, fall back to the older one.
    voronoi_atoms = voronoi_atoms_new if not force_map and num_arom < 7 else voronoi_atoms_old
    voronoi, _ = voronoi_atoms(
        cgbead_coords, heavyatom_coords, allatom_coords, mol, heavy_neighbor_of,
        geometry=geometry, bead_heavy_idx=bead_heavy_idx, with_cog=with_cog, ring_groups=ring_groups,
    )
    logger.debug("voronoi %s", voronoi)

    # Precompute, once per trial_comb, per-region atom counts and per-region
    # counts of bonds fully contained within that region: the double loop
    # below only ever needs these two counts per bead's region.
    region_size = Counter(voronoi.values())
    same_region_bond_count = Counter()
    for b0, b1 in bondlist:
        region0 = voronoi[heavy_index[b0]]
        region1 = voronoi[heavy_index[b1]]
        if region0 == region1:
            same_region_bond_count[region0] += 1

    for i in range(len(trial_comb)):
        cg_bead = trial_comb[i]
        region = voronoi[heavy_index[cg_bead]]
        num_atoms = region_size[region]
        num_bonds = same_region_bond_count[region]
        if num_bonds < num_atoms - 1 or num_atoms == 1:
            logger.debug("Error: Not all atoms in beads connected in %s", trial_comb)
            logger.debug("Error: %s < %s", num_bonds, num_atoms - 1)
            return False
    return True
