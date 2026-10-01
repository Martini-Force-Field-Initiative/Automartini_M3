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