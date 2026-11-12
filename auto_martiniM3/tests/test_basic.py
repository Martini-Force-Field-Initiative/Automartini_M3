"""
Basic sanity test for the auto_martini package.
"""
import filecmp
import os
from pathlib import Path
from rdkit import Chem
import pytest

import auto_martiniM3

dpath = Path(__file__).parent / "files"


def test_auto_martini_imported():
    """Sample test, will always pass so long as import statement worked"""
    import sys

    assert "auto_martiniM3" in sys.modules


@pytest.mark.parametrize(
    "smiles",
    [
        ("CC(=O)OC1=CC=CC=C1C(=O)O")
    ],
)
def test_connection_to_ALOGPS(smiles: str):
    logp = auto_martiniM3.topology.smi2alogps(False, smiles, None, "MOL", False, None, logp_file=None, trial=False)
    assert logp is not None

@pytest.mark.parametrize(
    "smiles,name,num_beads",
    [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "ASP", 5),
        ("CCC", "PRO", 1),
    ],
)

def test_auto_martini_run_smiles(smiles: str, name: str, num_beads: int):
    mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
    cg_mol = auto_martiniM3.solver.Cg_molecule(mol, smiles, name,_,_,_,_,_,_)
    assert len(cg_mol.cg_bead_names) == num_beads


@pytest.mark.parametrize(
    "sdf_file,name,num_beads", 
    [
        (dpath / "benzene.sdf", "BENZ", 3), 
        (dpath / "ibuprofen.sdf", "IBUP", 6)
    ],
)

def test_auto_martini_run_sdf(sdf_file: str, name:str, num_beads: int):
    mol = auto_martiniM3.topology.gen_molecule_sdf(str(sdf_file))
    smiles = str(Chem.MolToSmiles(mol, isomericSmiles=False))
    cg_mol = auto_martiniM3.solver.Cg_molecule(mol, smiles, name,None,None,None,None,None,None)
    assert len(cg_mol.cg_bead_names) == num_beads


@pytest.mark.parametrize("smiles", ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCC"])
def test_parallel_search_matches_sequential(smiles: str):
    """The bead search gives the same candidates, in the same order and with the
    same element types, on one process and on several."""
    mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
    data = auto_martiniM3.solver._MoleculeData(auto_martiniM3.solver.Cg_molecule._embed(mol))
    args = (data.molecule, data.conformer, data.heavy_atoms, data.heavy_atom_coords, data.atom_coords,
            data.ring_atoms, data.ring_atoms_flat, False)
    sequential, parallel = [
        ([[(type(a).__name__, int(a)) for a in comb] for comb in cg_beads], [repr(p) for p in bead_pos])
        for cg_beads, bead_pos in (auto_martiniM3.optimization.find_bead_pos(*args, nproc=n) for n in (1, 2))
    ]
    assert sequential == parallel

def _search_arguments(smiles):
    mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
    data = auto_martiniM3.solver._MoleculeData(auto_martiniM3.solver.Cg_molecule._embed(mol))
    return (data.molecule, data.conformer, data.heavy_atoms, data.heavy_atom_coords, data.atom_coords,
            data.ring_atoms, data.ring_atoms_flat)


def _typed(cg_beads, bead_pos):
    return [[(type(a).__name__, int(a)) for a in comb] for comb in cg_beads], [repr(p) for p in bead_pos]


# CCC: 3 heavy atoms, the 1-bead level is listed twice; the others have rings and several levels.
@pytest.mark.parametrize("smiles", ["CCC", "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                                    "c1ccc2cc3ccccc3cc2c1"])
def test_branch_and_bound_matches_exhaustive(smiles: str):
    """The branch-and-bound search gives find_bead_pos()'s candidates, in the same
    order and with the same element types, and the same largest bead count."""
    from auto_martiniM3.optimization import branch_and_bound
    args = _search_arguments(smiles)
    exhaustive = auto_martiniM3.optimization.find_bead_pos(*args, False, nproc=1)
    lazy = branch_and_bound.LazyCandidates(branch_and_bound.BeadSearch(*args[:5], args[6], False))
    lazy.generate(len(exhaustive[0]) + 1)
    assert lazy.exhausted
    assert _typed(lazy.cg_beads, lazy.bead_pos) == _typed(*exhaustive)
    largest = max(len(c) for c in exhaustive[0])
    assert lazy.reaches(largest) and not lazy.reaches(largest + 1)


@pytest.mark.parametrize("lazy_candidates,max_frontier", [(1000, 6000000), (3, 6000000), (1000, 5)])
def test_bead_candidates_hand_over(monkeypatch, lazy_candidates: int, max_frontier: int):
    """Past LAZY_CANDIDATES candidates, or with a frontier past MAX_FRONTIER entries,
    bead_candidates() goes on with find_bead_pos()'s list: same candidates throughout."""
    from auto_martiniM3.optimization import branch_and_bound
    monkeypatch.setattr(branch_and_bound, "LAZY_CANDIDATES", lazy_candidates)
    monkeypatch.setattr(branch_and_bound, "MAX_FRONTIER", max_frontier)
    args = _search_arguments("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    expected = _typed(*auto_martiniM3.optimization.find_bead_pos(*args, False, nproc=1))
    candidates = auto_martiniM3.optimization.bead_candidates(*args, method="branch-and-bound", nproc=1)
    got = [candidates[i] for i in range(len(expected[0]))]
    assert not candidates.more_than(len(expected[0])) and len(candidates) == len(expected[0])
    assert _typed([c for c, _ in got], [p for _, p in got]) == expected


def test_search_methods_same_topology(tmp_path, monkeypatch):
    """A whole run gives the same files with either bead search."""
    monkeypatch.chdir(tmp_path)
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    files = {}
    for method in ("branch-and-bound", "exhaustive"):
        mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
        cg_mol = auto_martiniM3.solver.Cg_molecule(mol, smiles, "IBUP", False, method + ".itp", "", False, None,
                                                   True, nproc=1, search=method)
        cg_mol.output_cg(method + ".gro")
        files[method] = [(tmp_path / (method + ext)).read_text().replace(method, "X") for ext in (".itp", ".gro")]
    assert files["branch-and-bound"] == files["exhaustive"]
