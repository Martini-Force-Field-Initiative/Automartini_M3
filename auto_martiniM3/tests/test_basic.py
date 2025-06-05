"""
Basic sanity test for the auto_martini package.
"""
import filecmp
import os
from pathlib import Path
from rdkit import Chem
import pytest

import auto_martiniM3

dpath = Path("tests/files")


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