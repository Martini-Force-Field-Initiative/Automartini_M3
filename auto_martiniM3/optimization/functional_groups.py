"""Functional-group detection and the mapping constraints built on it.

identify_functional_groups() is Ertl's functional-group finder (RDKit
contrib, Hall & Godin), used here to check that a candidate bead mapping
keeps each functional group inside a single bead (functional_groups_ok())
and doesn't crowd more than two aromatic atoms into one bead (max2arperbead()).
"""

from ..common import *

logger = logging.getLogger(__name__)


def merge(mol, marked, aset):
    #  Original authors: Richard Hall and Guillaume Godin
    #  This file is part of the RDKit.
    #  The contents are covered by the terms of the BSD license
    #  which is included in the file license.txt, found at the root
    #  of the RDKit source tree.
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)


def identify_functional_groups(mol):
    # atoms connected by non-aromatic double or triple bond to any heteroatom
    PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
    # atoms in non-aromatic carbon-carbon double or triple bonds
    PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
    # acetal carbons, i.e. sp3 carbons connected to two or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
    PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
    # all atoms in oxirane, aziridine and thiirane rings
    PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')
    # the bridge between two aromatic cycles
    PATT_BRIDGE_AROMATIC = Chem.MolFromSmarts("[x;!x2]")

    PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC, PATT_BRIDGE_AROMATIC)

    marked = set()
    # mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    # mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    # merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

    # extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type', 'type_atomIds'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        type_atoms = g.union(uca)
        ifgs.append(
            ifg(atomIds=tuple(sorted(g)),
                atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
                type=Chem.MolFragmentToSmiles(mol, type_atoms, canonical=True),
                type_atomIds=tuple(sorted(type_atoms)))
        )
    return ifgs


def functional_groups_ok(atom_partitioning,molecule,ringatoms):
    """
    Checking if functional groups are conserved in distinctive bead, within atom number per bead limit.
    """

    fgs = identify_functional_groups(molecule)

    bead_atoms={}
    for at, bead in atom_partitioning.items():
        if bead not in bead_atoms:
            bead_atoms[bead] = []
        bead_atoms[bead].append(at)

    group_found = []
    for ix, fg in enumerate(fgs):
        gr_f = False

        for bead, atoms in bead_atoms.items():
            if set(fg.type_atomIds).issubset(atoms) or len(fg.type_atomIds)>=3: #do not change!!!! better symmetry if len >=3
                gr_f = True
                break
        group_found.append(gr_f)

    # Check if at least 50% of elements in group_found are True
    if group_found.count(True) >= len(group_found) / 2 :
        return True
    else:
        return False


def max2arperbead(atom_partitioning, ringatoms):
    """
    Checking the number of aromatic atoms in a bead and returning False if it's more than 2.
    """
    bead_atoms = {}
    for at, bead in atom_partitioning.items():
        if bead not in bead_atoms:
            bead_atoms[bead] = []
        bead_atoms[bead].append(at)

    # Convert ringatoms to a set
    ringatoms_set = set(atom for sublist in ringatoms for atom in sublist)
    for bead,atoms in bead_atoms.items():
        ring_atom_count = sum(1 for atom in atoms if atom in ringatoms_set)
        if ring_atom_count > 2:
            return False
    return True
