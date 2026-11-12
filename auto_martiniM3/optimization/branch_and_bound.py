"""Branch-and-bound bead search: find_bead_pos()'s candidates, best first and on demand.

find_bead_pos() scores every valid combination of bead positions, keeps the
connected ones and sorts them all by energy, while the solver usually needs only
the first one or the first few. LazyCandidates explores the same enumeration tree
(same levels, same pruning rules, same early stop between levels) best first:
each partial combination gets a lower bound of the energy any of its completions
can reach, and the candidates come out in increasing (energy, level, enumeration
order) -- the order of the stable sort over find_bead_pos()'s list -- while only
the subtrees whose bound beats the next candidate are ever opened.

Energy (eval_gaussian_interac) = sum over beads of (offset + in_gaussian)   independent terms
                                + sum over bead pairs of their overlap       >= 0
                                + 0.28 * mass of the atoms no bead lumps     >= 0
Lower bound of a partial combination (positions chosen, m more to pick among later positions):
    the exact single terms and pair overlaps of the chosen beads
  + the m smallest single terms among the later positions still allowed (bonded / terminal-partner rules)
  + 0.28 * mass of the atoms that neither the chosen beads nor any later position can lump.
Leaves are scored with eval_gaussian_interac() itself, so candidate energies are exactly the same floats.

bead_candidates() is the entry point the solver uses. It falls back on find_bead_pos(),
which gives the same candidates in the same order, whenever the lazy search would not
pay off: with -v (the log lists every trial), past LAZY_CANDIDATES candidates (the solver
then needs most of the list, which find_bead_pos() computes faster, on all CPUs), when
the frontier grows past MAX_FRONTIER entries (memory), and for the cases find_bead_pos()
handles specially (0, 1 or more than 50 heavy atoms, no candidate at all).
"""

import heapq
import logging
import os

from .energy import eval_gaussian_interac, read_bead_params
from . import energy, partition, search as search_module
from .search import BeadSearch, _gc_paused, all_atoms_in_beads_connected, find_bead_pos

logger = logging.getLogger(__name__)

SEARCH_ENV = "AUTOMARTINI_SEARCH"
SEARCH_METHODS = ("branch-and-bound", "exhaustive")
# Past this many candidates, the rest of the list comes from find_bead_pos().
LAZY_CANDIDATES = 1000
# Frontier entries (about 500 bytes each) before handing over to find_bead_pos().
MAX_FRONTIER = 6000000

EPS = 1e-7  # bounds are lowered by this much, so that float rounding never puts one above a true energy

# Kinds of frontier entries: a partial combination; a full one with its bound; a full one with
# its exact energy, connectivity not yet tested; a full one already known to be connected.
INTERNAL, LEAF_BOUND, LEAF_EXACT, LEAF_CONNECTED = 0, 1, 2, 3


class FrontierTooLarge(Exception):
    """The branch-and-bound frontier went past MAX_FRONTIER entries."""


class _MassTable:
    """Summed mass of the atoms of a bitmask, by byte lookups."""

    def __init__(self, masses):
        self.tables = []
        for b in range(0, len(masses), 8):
            chunk = masses[b:b + 8]
            self.tables.append([sum(chunk[i] for i in range(len(chunk)) if v >> i & 1) for v in range(256)])

    def __call__(self, mask):
        total, b = 0.0, 0
        while mask:
            total += self.tables[b][mask & 255]
            mask >>= 8
            b += 1
        return total


class BranchAndBound:
    """Best-first search over the enumeration tree of BeadSearch.combinations().

    A node is (positions, single_sum, overlap_sum, cover, blocked, partners): the
    positions chosen so far (increasing), their summed single terms and pair
    overlaps, and bitmasks of the atoms they lump, the positions they rule out
    (bonded to one of them) and the atoms their terminal positions hang off.
    """

    def __init__(self, search, max_frontier=None):
        self.search = search
        self.max_frontier = max_frontier
        params = read_bead_params()
        atoms = search.list_heavy_atoms
        n = self.n = len(atoms)
        cache = search.energy_cache
        ring = search.ringatoms_flat
        self.single, self.lump = [], []
        for atom in atoms:
            offset = params["offset_bd_aromatic_weight"] if atom in ring else params["offset_bd_weight"]
            weight, lumped = cache.in_gaussian(atom)
            self.single.append(offset + weight)
            mask = 0
            for lumped_atom in lumped:
                mask |= 1 << int(lumped_atom)
            self.lump.append(mask)
        self.overlap = [[cache.overlap(atoms[i], atoms[j]) if i < j else 0.0 for j in range(n)] for i in range(n)]
        self.all_mask = (1 << search.conformer.GetNumAtoms()) - 1
        self.mass = _MassTable(cache.masses)
        self.lonely = params["lonely_atom_penalize"]
        self.suffix_cover = [0] * (n + 1)  # atoms some position >= k can lump
        for k in range(n - 1, -1, -1):
            self.suffix_cover[k] = self.suffix_cover[k + 1] | self.lump[k]
        self.adjacency_mask = [sum(1 << q for q in search.adjacency[p]) for p in range(n)]
        self.partner = search.terminal_partner
        self.by_single = sorted(range(n), key=lambda p: self.single[p])
        self.nodes_opened = 0
        self.leaves_scored = 0
        self.connectivity_tests = 0
        self.max_heap = 0

    def _bound(self, level, positions, single_sum, overlap_sum, cover, blocked, partners):
        """Lower bound of the energy of any completion of positions to level beads, or None if there is none."""
        m = level - len(positions)
        last = positions[-1] if positions else -1
        rest, found = 0.0, 0
        if m:
            for p in self.by_single:
                if p <= last or blocked >> p & 1:
                    continue
                partner = self.partner[p]
                if partner is not None and partners >> partner & 1:
                    continue
                rest += self.single[p]
                found += 1
                if found == m:
                    break
            if found < m:
                return None
        coverable = cover if m == 0 else cover | self.suffix_cover[last + 1]
        return single_sum + overlap_sum + rest + self.lonely * self.mass(self.all_mask & ~coverable)

    def _children(self, level, node):
        """(bound, child) of each child of node that has a completion: the same
        positions, in the same order, as _valid_bead_combinations() tries."""
        positions, single_sum, overlap_sum, cover, blocked, partners = node
        last = positions[-1] if positions else -1
        last_start = self.n - (level - len(positions))
        for q in range(last + 1, last_start + 1):
            if blocked >> q & 1:
                continue
            partner = self.partner[q]
            if partner is not None and partners >> partner & 1:
                continue
            child = (positions + (q,), single_sum + self.single[q],
                     overlap_sum + sum(self.overlap[c][q] for c in positions), cover | self.lump[q],
                     blocked | self.adjacency_mask[q], partners | (1 << partner if partner is not None else 0))
            bound = self._bound(level, *child)
            if bound is not None:
                yield bound, child

    def push_children(self, heap, level_index, level, node=((), 0.0, 0.0, 0, 0, 0)):
        """Add the children of node (by default the root) to the frontier heap.

        Entries are (key, level index, positions, kind, payload): ties on the key
        go to the lower level, then to the earlier combination in enumeration order.
        """
        self.nodes_opened += 1
        for bound, child in self._children(level, node):
            kind = LEAF_BOUND if len(child[0]) == level else INTERNAL
            heapq.heappush(heap, (bound - EPS, level_index, child[0], kind, child))
        if len(heap) > self.max_heap:
            self.max_heap = len(heap)
            if self.max_frontier is not None and self.max_heap > self.max_frontier:
                raise FrontierTooLarge(self.max_heap)

    def step(self, heap, levels):
        """Process the best frontier entry. Return (energy, level index, positions, trial_comb)
        if it is a connected combination, else None."""
        key, level_index, positions, kind, payload = heapq.heappop(heap)
        if kind == INTERNAL:
            self.push_children(heap, level_index, levels[level_index], payload)
        elif kind == LEAF_BOUND:
            search = self.search
            trial_comb = search.trial_comb(positions)
            self.leaves_scored += 1
            trial_ene = eval_gaussian_interac(search.molecule, search.conformer, trial_comb, search.ringatoms_flat,
                                              cache=search.energy_cache)
            heapq.heappush(heap, (trial_ene, level_index, positions, LEAF_EXACT, trial_comb))
        elif kind == LEAF_EXACT:
            search = self.search
            self.connectivity_tests += 1
            if all_atoms_in_beads_connected(
                payload, search.heavyatom_coords, search.list_heavy_atoms, search.list_bonds, search.molecule,
                search.allatom_coords, search.force_map, heavy_index=search.heavy_index, num_arom=search.num_arom,
                heavy_neighbor_of=search.heavy_neighbor_of, geometry=search.geometry,
                ring_groups=search.ring_groups,
            ):
                return key, level_index, positions, payload
        return None


def _levels(num_heavy_atoms):
    """find_bead_pos()'s bead-count levels (the 1-bead level twice below 4 heavy atoms)."""
    min_beads, max_beads = int(num_heavy_atoms / 4.0), int(num_heavy_atoms / 2.0)
    return [max(num_beads, 1) for num_beads in range(min_beads, max_beads + 1)]


class LazyCandidates:
    """find_bead_pos()'s (cg_beads, bead_pos) list for one molecule, built on demand in the same order.

    Levels are explored in turn, each only until it is known whether it improves
    the best energy so far, with find_bead_pos()'s early stop; then all explored
    levels are merged into one best-first search, which gives the candidates in
    (energy, level, enumeration) order. May raise FrontierTooLarge.
    """

    def __init__(self, search, max_frontier=None):
        self.search = search
        self.bb = BranchAndBound(search, max_frontier)
        self.levels, heaps, found = [], [], []
        best_ene, best_comb, last_best_comb = 1e6, [], []
        for level_index, level in enumerate(_levels(len(search.list_heavy_atoms))):
            self.levels.append(level)
            heap = []
            self.bb.push_children(heap, level_index, level)
            first = None
            while heap and heap[0][0] < best_ene:
                first = self.bb.step(heap, self.levels)
                if first is not None:
                    break
            heaps.append(heap)
            if first is not None:
                found.append(first)
                best_ene, best_comb = first[0], sorted(first[3])
            if last_best_comb == best_comb:
                break
            last_best_comb = best_comb
        self._level_has_candidate = {level_index: True for _, level_index, _, _ in found}
        # One frontier for every explored level; the leaves found above go back in as known to be connected.
        self.heap = [entry for heap in heaps for entry in heap]
        for trial_ene, level_index, positions, trial_comb in found:
            self.heap.append((trial_ene, level_index, positions, LEAF_CONNECTED, trial_comb))
        heapq.heapify(self.heap)
        self.cg_beads, self.bead_pos = [], []
        self.exhausted = False

    def _next(self):
        """Add the next candidate to cg_beads and bead_pos; False if there is none left."""
        while self.heap:
            if self.heap[0][3] == LEAF_CONNECTED:
                trial_ene, level_index, positions, _, trial_comb = heapq.heappop(self.heap)
                hit = trial_ene, level_index, positions, trial_comb
            else:
                hit = self.bb.step(self.heap, self.levels)
            if hit is not None:
                _, level_index, positions, trial_comb = hit
                self._level_has_candidate[level_index] = True
                self.cg_beads.append(trial_comb)
                self.bead_pos.append(self.search.bead_positions(positions, trial_comb))
                return True
        self.exhausted = True
        return False

    def generate(self, count):
        """Build the list up to count candidates, or all of them if there are fewer."""
        while len(self.cg_beads) < count and not self.exhausted:
            self._next()

    def reaches(self, num_beads):
        """True if some candidate has at least num_beads beads."""
        deeper = [i for i, level in enumerate(self.levels) if level >= num_beads]
        if any(self._level_has_candidate.get(i) for i in deeper):
            return True
        search = self.search
        for i in deeper:
            if self._level_has_candidate.get(i) is None:
                self._level_has_candidate[i] = next(
                    search.connected_combinations(search.combinations(self.levels[i])), None) is not None
            if self._level_has_candidate[i]:
                return True
        return False


def _frontier_message(error):
    return "frontier of more than {} entries (MAX_FRONTIER)".format(error.args[0] - 1)


def _search_logs():
    """True if the bead search logs every trial (INFO level or below)."""
    return any(log.isEnabledFor(logging.INFO) for log in (search_module.logger, energy.logger, partition.logger))


def search_method(method=None):
    """The bead search to use: method if given, else the AUTOMARTINI_SEARCH environment
    variable, else "branch-and-bound"."""
    if method is None:
        method = os.environ.get(SEARCH_ENV, "").strip().lower() or SEARCH_METHODS[0]
    if method not in SEARCH_METHODS:
        raise ValueError("bead search must be one of {}, not {!r}".format(", ".join(SEARCH_METHODS), method))
    return method


class _ExhaustiveCandidates:
    """find_bead_pos()'s list, behind the interface of bead_candidates()."""

    def __init__(self, cg_beads, bead_pos):
        self.cg_beads, self.bead_pos = cg_beads, bead_pos

    def __getitem__(self, i):
        return self.cg_beads[i], self.bead_pos[i]

    def more_than(self, count):
        return len(self.cg_beads) > count

    def __len__(self):
        return len(self.cg_beads)

    def reaches(self, num_beads):
        return max((len(c) for c in self.cg_beads), default=0) >= num_beads


class _LazyThenExhaustive:
    """LazyCandidates up to LAZY_CANDIDATES candidates, then find_bead_pos()'s list."""

    def __init__(self, lazy, exhaustive):
        self.lazy, self.exhaustive, self.full = lazy, exhaustive, None

    def _ensure(self, count):
        """Make the first count candidates (or all, if fewer) available."""
        lazy = self.lazy
        if lazy is None or len(lazy.cg_beads) >= count or lazy.exhausted:
            return
        if count > LAZY_CANDIDATES:
            self._hand_over("the solver needs more than {} candidates".format(LAZY_CANDIDATES))
            return
        try:
            with _gc_paused():
                lazy.generate(count)
        except FrontierTooLarge as error:
            self._hand_over(_frontier_message(error), logging.WARNING)

    def _hand_over(self, reason, level=logging.INFO):
        logger.log(level, "Branch and bound hands over to the exhaustive search: %s", reason)
        self.lazy = None  # frees the frontier before the exhaustive search
        self.full = self.exhaustive()

    def _list(self):
        return self.full if self.lazy is None else self.lazy

    def __getitem__(self, i):
        self._ensure(i + 1)
        candidates = self._list()
        return candidates.cg_beads[i], candidates.bead_pos[i]

    def more_than(self, count):
        self._ensure(count + 1)
        return len(self._list().cg_beads) > count

    def __len__(self):
        self._ensure(LAZY_CANDIDATES)
        if self.lazy is not None and not self.lazy.exhausted:
            self._hand_over("the solver needs the number of candidates")
        return len(self._list().cg_beads)

    def reaches(self, num_beads):
        if self.lazy is None:
            return self.full.reaches(num_beads)
        with _gc_paused():
            return self.lazy.reaches(num_beads)


def bead_candidates(molecule, conformer, list_heavy_atoms, heavyatom_coords, allatom_coords, ring_atoms,
                    ringatoms_flat, method=None, nproc=None):
    """find_bead_pos()'s candidates (force_map False), best first, as an object with:
    c[i] -> (cg_beads, bead_pos) of candidate i, c.more_than(n), len(c), and c.reaches(b),
    True if some candidate has at least b beads.

    method: "branch-and-bound" (default) gives them on demand, "exhaustive" computes
    them all with find_bead_pos(), see search_method(). nproc: processes for find_bead_pos().
    """

    def exhaustive():
        return _ExhaustiveCandidates(*find_bead_pos(molecule, conformer, list_heavy_atoms, heavyatom_coords,
                                                    allatom_coords, ring_atoms, ringatoms_flat, False,
                                                    nproc=nproc))

    if (search_method(method) == "exhaustive" or not 1 < len(list_heavy_atoms) <= 50 or _search_logs()):
        return exhaustive()
    try:
        with _gc_paused():
            lazy = LazyCandidates(BeadSearch(molecule, conformer, list_heavy_atoms, heavyatom_coords,
                                             allatom_coords, ringatoms_flat, False), MAX_FRONTIER)
            lazy.generate(1)
    except FrontierTooLarge as error:
        logger.warning("Branch and bound hands over to the exhaustive search: %s", _frontier_message(error))
        return exhaustive()
    if not lazy.cg_beads:  # no candidate: find_bead_pos() handles that case its own way
        return exhaustive()
    return _LazyThenExhaustive(lazy, exhaustive)
