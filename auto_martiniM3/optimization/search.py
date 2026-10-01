"""Combinatorial CG bead-placement search.

find_bead_pos() is the entry point: it exhaustively tries every valid
combination of heavy atoms as bead headliners (check_beads() filters out
combinations that can't possibly be valid; all_atoms_in_beads_connected()
rejects combinations whose resulting bead partition isn't fully connected),
scores each surviving combination with the Gaussian objective function, and
returns every valid combination sorted by score. BeadSearch holds the data
computed once per molecule that all those trials share.
"""

import concurrent.futures
import contextlib
import ctypes
import gc
import multiprocessing
import signal

from rdkit.Geometry import Point3D

from ..common import *
from .. import topology
from . import energy, partition
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


def _valid_bead_combinations(num_atoms, num_beads, adjacency, terminal_partner, prefix=(), depth=None):
    """Yield, as tuples of positions in 0..num_atoms-1, exactly the num_beads-sized
    combinations that check_beads() accepts, in itertools.combinations order.

    check_beads() rejects a combination as soon as any two of its atoms are
    bonded (whether or not they share a ring, one branch or the other rejects
    it), or two of its terminal atoms hang off the same neighbor. Both rules are
    pairwise, so a depth-first search that picks positions in increasing order
    can prune a partial combination the moment it breaks one -- instead of
    building every combination and rejecting ~97% of them afterwards.

    prefix: only the combinations starting with these positions (itself a partial
    combination this function yields). depth: stop at partial combinations of
    that length. Together they split the search into subtrees that, taken in
    order, give back the full sequence.
    """
    target = num_beads if depth is None else depth
    combo = list(prefix)
    blocked = [0] * num_atoms
    partners_in_use = Counter()
    for p in prefix:
        for q in adjacency[p]:
            blocked[q] += 1
        if terminal_partner[p] is not None:
            partners_in_use[terminal_partner[p]] += 1

    def extend(start):
        if len(combo) == target:
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

    yield from extend(prefix[-1] + 1 if prefix else 0)


def _rebuild_bead_search(molecule, conformer_id, atom_xyz, *args):
    """Unpickling helper of BeadSearch: puts back the exact atom coordinates."""
    conformer = molecule.GetConformer(conformer_id)
    for i, xyz in enumerate(atom_xyz):
        conformer.SetAtomPosition(i, Point3D(*xyz))
    return BeadSearch(molecule, conformer, *args)


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
        # Per position in list_heavy_atoms: the atom as the np.int64 trial_comb holds,
        # and its coordinates. Combinations list positions in increasing order, so with
        # heavy atoms in increasing index order (as get_atoms() lists them) a
        # trial_comb is already sorted.
        self._heavy_atom_items = list(self.heavy_atoms_array)
        self._heavy_xyz = [self.atom_xyz[int(atom)] for atom in list_heavy_atoms]
        self._heavy_atoms_increasing = all(a < b for a, b in zip(list_heavy_atoms, list_heavy_atoms[1:]))

    def __reduce__(self):
        # Rebuilt from its constructor arguments in another process. RDKit pickles
        # conformer coordinates in single precision, so the exact ones travel along.
        return (_rebuild_bead_search, (self.molecule, self.conformer.GetId(), self.atom_xyz, self.list_heavy_atoms,
                                       self.heavyatom_coords, self.allatom_coords, self.ringatoms_flat,
                                       self.force_map))

    def combinations(self, num_beads, prefix=(), depth=None):
        """Positions (in list_heavy_atoms) of every num_beads-bead combination
        check_beads() accepts, in itertools.combinations order -- or only the
        subtree under prefix, or partial combinations of length depth (see
        _valid_bead_combinations)."""
        return _valid_bead_combinations(len(self.list_heavy_atoms), num_beads, self.adjacency,
                                        self.terminal_partner, prefix, depth)

    def connected_combinations(self, combinations):
        """Score each combination of positions and yield, in the same order,
        (positions, trial_comb, energy) for those whose beads would hold connected atoms."""
        for positions in combinations:
            trial_comb = self.trial_comb(positions)
            trial_ene = eval_gaussian_interac(self.molecule, self.conformer, trial_comb, self.ringatoms_flat,
                                              cache=self.energy_cache)
            logger.info("; %s %s", trial_comb, trial_ene)
            if all_atoms_in_beads_connected(
                trial_comb, self.heavyatom_coords, self.list_heavy_atoms, self.list_bonds, self.molecule,
                self.allatom_coords, self.force_map, heavy_index=self.heavy_index, num_arom=self.num_arom,
                heavy_neighbor_of=self.heavy_neighbor_of, geometry=self.geometry, ring_groups=self.ring_groups,
            ):
                yield positions, trial_comb, trial_ene

    def trial_comb(self, positions):
        """Heavy atoms at these positions, as a list of np.int64."""
        return [self._heavy_atom_items[p] for p in positions]

    def bead_positions(self, positions, trial_comb):
        """Coordinates of the beads of trial_comb (at these increasing positions), in sorted atom order."""
        if self._heavy_atoms_increasing:
            return [list(self._heavy_xyz[p]) for p in positions]
        return [list(self.atom_xyz[int(atom)]) for atom in sorted(trial_comb)]

    def scored_level(self, num_beads):
        """(trial_comb, bead positions, energy) of every connected num_beads-bead
        combination, in enumeration order."""
        for positions, trial_comb, trial_ene in self.connected_combinations(self.combinations(num_beads)):
            yield trial_comb, self.bead_positions(positions, trial_comb), trial_ene

    def unpack(self, positions, energies):
        """Same as scored_level() for the part a worker scored: positions is its
        (combinations x beads) array, energies the matching energies."""
        for row, trial_ene in zip(positions.tolist(), energies.tolist()):
            trial_comb = self.trial_comb(row)
            yield trial_comb, self.bead_positions(row, trial_comb), trial_ene


# ---------------------------------------------------------------------------
# Parallel search
#
# The trial combinations are independent, so each bead-count level is split
# into subtrees of the enumeration (by combination prefix), scored in worker
# processes, and put back together in enumeration order: the result is the
# same, element for element, as the sequential search. Processes, not threads:
# the search is pure Python, which threads can't run in parallel.
# ---------------------------------------------------------------------------

NPROC_ENV = "AUTOMARTINI_NPROC"
# Automatic mode stays sequential below this many heavy atoms: the whole search
# then takes less time than starting the worker processes.
PARALLEL_MIN_HEAVY_ATOMS = 20
# Subtrees per worker, so that uneven subtrees still keep every worker busy.
_TASKS_PER_WORKER = 8

_worker_search = None  # the BeadSearch of the current molecule, in a worker process


def available_cpus():
    """Number of CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def worker_count(nproc, num_heavy_atoms):
    """Number of processes for the bead search; 1 means sequential.

    nproc: requested number, or None for automatic, in which case the
    AUTOMARTINI_NPROC environment variable is used if set (a number, or "auto").
    Automatic mode uses every available CPU for molecules of at least
    PARALLEL_MIN_HEAVY_ATOMS heavy atoms, on Linux only, and not when already
    running in a worker process of the caller (its own pool of molecules). The
    search stays sequential anyway when it logs at INFO level or below (the
    per-combination log keeps its order) and inside a daemonic process, which
    can't start workers.
    """
    return _pool_setup(nproc, num_heavy_atoms)[0]


def _pool_setup(nproc, num_heavy_atoms):
    """(number of processes, multiprocessing context) for the bead search, see worker_count().

    Automatic mode starts its workers with fork: they then don't re-import the
    caller's main script, which "spawn" (the default on macOS and Windows) does
    and which fails for a script without an if __name__ == "__main__" guard.
    An explicit nproc uses the platform's default start method.
    """
    if nproc is None:
        requested = os.environ.get(NPROC_ENV, "").strip().lower()
        if requested not in ("", "auto"):
            try:
                nproc = int(requested)
            except ValueError:
                raise ValueError(f"{NPROC_ENV} must be a number of processes or 'auto', not {requested!r}")
    context = None
    if nproc is None:
        if (num_heavy_atoms < PARALLEL_MIN_HEAVY_ATOMS or not sys.platform.startswith("linux")
                or multiprocessing.parent_process() is not None):
            return 1, None
        nproc, context = available_cpus(), multiprocessing.get_context("fork")
    search_logs = any(log.isEnabledFor(logging.INFO) for log in (logger, energy.logger, partition.logger))
    if nproc <= 1 or search_logs or multiprocessing.current_process().daemon:
        return 1, None
    return nproc, context


def _start_worker(search):
    """Pool initializer: keep the molecule's BeadSearch for the tasks of this worker."""
    global _worker_search
    _worker_search = search
    gc.enable()  # a forked worker inherits the parent's paused collector (_gc_paused)
    if sys.platform.startswith("linux"):
        # Terminate with the parent even when it is killed outright (e.g. by a time limit),
        # instead of lingering as an orphan. 1 = PR_SET_PDEATHSIG.
        try:
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        except (OSError, AttributeError):
            pass


def _connected_in_subtree(task):
    """Worker task: positions and energies of the connected combinations of one
    subtree, in enumeration order. An exception (SystemExit included) is
    returned rather than raised, for the parent to raise it again."""
    num_beads, prefix = task
    try:
        search = _worker_search
        found = [(positions, trial_ene) for positions, _, trial_ene
                 in search.connected_combinations(search.combinations(num_beads, prefix))]
        # Two arrays pickle far faster than a list of tuples; float64 keeps the energies exact.
        positions = np.array([p for p, _ in found], dtype=np.int32).reshape(len(found), num_beads)
        energies = np.array([e for _, e in found], dtype=np.float64)
        return positions, energies
    except BaseException as error:
        return error


def _subtree_prefixes(search, num_beads, workers):
    """Split one bead-count level into subtrees, in enumeration order: the
    shortest prefixes that give at least _TASKS_PER_WORKER subtrees per worker."""
    depth = 1
    prefixes = list(search.combinations(num_beads, depth=depth))
    while len(prefixes) < _TASKS_PER_WORKER * workers and depth < num_beads - 1:
        depth += 1
        prefixes = list(search.combinations(num_beads, depth=depth))
    return prefixes


class _ParallelLevels:
    """Bead-count levels scored by a process pool. While the parent goes
    through the results of one level, the workers already score the next one
    (dropped if the search stops before it). Submitted subtrees are never
    cancelled: with Python 3.8 cancelling pending futures can make the pool's
    shutdown wait forever, so an unneeded level is scored to the end instead."""

    def __init__(self, search, pool, workers, levels):
        self.search, self.pool, self.workers, self.levels = search, pool, workers, levels
        self.futures = {}  # level index -> futures of its subtrees, in order

    def _submit(self, index):
        if index < len(self.levels) and index not in self.futures:
            num_beads = self.levels[index]
            self.futures[index] = [self.pool.submit(_connected_in_subtree, (num_beads, prefix))
                                   for prefix in _subtree_prefixes(self.search, num_beads, self.workers)]

    def scored_level(self, index):
        """Same as search.scored_level(levels[index])."""
        self._submit(index)
        self._submit(index + 1)
        for future in self.futures[index]:
            result = future.result()
            if isinstance(result, BaseException):
                raise result
            yield from self.search.unpack(*result)
        del self.futures[index]


@contextlib.contextmanager
def _gc_paused():
    """Pause the cyclic garbage collector. The search piles up millions of small
    lists, none in a reference cycle, and the collector would otherwise go over
    all of them again and again as they accumulate."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


@contextlib.contextmanager
def _level_scorer(search, levels, workers, context):
    """Function giving the scored candidates of levels[index], computed in this
    process or, with more than one worker, in a process pool."""
    if workers <= 1:
        yield lambda index: search.scored_level(levels[index])
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_start_worker,
                                                initargs=(search,)) as pool:
        yield _ParallelLevels(search, pool, workers, levels).scored_level


def find_bead_pos(
    molecule, conformer, list_heavy_atoms, heavyatom_coords, allatom_coords, ring_atoms, ringatoms_flat, force_map,
    nproc=None,
):
    """Try out all possible combinations of CG beads up to threshold number of beads per atom. Find
    arrangement with best energy score. Return all possible arrangements sorted by energy score.

    nproc: number of processes for the search (None: automatic, see worker_count())."""

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

    # In Martini 3 a bead covers 2 to 4 heavy atoms. With fewer than 4 heavy atoms
    # min_beads is 0 and that level runs as 1 bead, so the 1-bead level is scored
    # twice and its candidates are listed twice.
    min_beads = int(len(list_heavy_atoms) / 4.0)
    max_beads = int(len(list_heavy_atoms) / 2.0)
    levels = [max(num_beads, 1) for num_beads in range(min_beads, max_beads + 1)]

    candidates = []  # [trial_comb, bead positions, energy] of every connected combination
    best_trial_comb, ene_best_trial = [], 1e6
    last_best_trial_comb = []
    workers, context = _pool_setup(nproc, len(list_heavy_atoms))
    with _gc_paused(), _level_scorer(search, levels, workers, context) as scored_level:
        for index in range(len(levels)):
            for trial_comb, bead_pos, trial_ene in scored_level(index):
                if trial_ene < ene_best_trial:
                    ene_best_trial = trial_ene
                    best_trial_comb = sorted(trial_comb)
                candidates.append([trial_comb, bead_pos, trial_ene])

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
