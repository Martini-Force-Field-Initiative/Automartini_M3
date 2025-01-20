"""
Basic sanity test for the auto_martini package.
"""
import filecmp
import os
from pathlib import Path

import pytest

import auto_martiniM3

dpath = Path("auto_martiniM3/tests/files")


def test_auto_martini_imported():
    """Sample test, will always pass so long as import statement worked"""
    import sys

    assert "auto_martiniM3" in sys.modules


"""@pytest.mark.parametrize(
    "sdf_file,num_beads", [(dpath / "benzene.sdf", 3), (dpath / "ibuprofen.sdf", 6)]
)
def test_auto_martini_run_sdf(sdf_file: str, num_beads: int):
    mol = auto_martiniM3.topology.gen_molecule_sdf(str(sdf_file))
    cg_mol = auto_martiniM3.solver.Cg_molecule(mol,"MOL")
    assert len(cg_mol.cg_bead_names) == num_beads"""

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
    "smiles,top_file,name,num_beads",
    [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "valid_ASP.top", "ASP", 5),
        ("CCC", "valid_PRO.top", "PRO", 1),
    ],
)

def test_auto_martini_run_smiles(smiles: str, top_file: Path, name: str, num_beads: int):
    mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
    cg_mol = auto_martiniM3.solver.Cg_molecule(mol, smiles, name,_,_,_,_,_,_)
    # assert filecmp.cmp(dpath / top_file, "mol.top")
    assert len(cg_mol.cg_bead_names) == num_beads
