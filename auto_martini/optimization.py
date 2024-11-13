"""
Created on March 13, 2019 by Andrew Abi-Mansour

Updated on November 8, 2024 by Magdalena Szczuka

This is the::

	     _   _   _ _____ ___    __  __    _    ____ _____ ___ _   _ ___ 
	    / \ | | | |_   _/ _ \  |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|
	   / _ \| | | | | || | | | | |\/| | / _ \ | |_) || |  | ||  \| || | 
	  / ___ \ |_| | | || |_| | | |  | |/ ___ \|  _ < | |  | || |\  || | 
	 /_/   \_\___/  |_| \___/  |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|                                                            
                                                                 
Tool for automatic MARTINI mapping and parametrization of small organic molecules

Developers::

	Tristan BEREAU (bereau at mpip-mainz.mpg.de)
	Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
	Andrew Abi-Mansour (andrew.gaam at gmail.com)
    Magdalena Szczuka (magdalena.szczuka@univ-tlse3.fr)

AUTO_MARTINI is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
"""

from sys import exit

import numpy as np
from collections import namedtuple
from .common import *
from . import topology # AutoM3 change

logger = logging.getLogger(__name__)


def read_bead_params():
    """Returns bead parameter dictionary
    CG Bead vdw radius (in Angstroem)"""
    bead_params = dict()
    bead_params["rvdw"] = 4.7 / 2.0     # sigma for non-ring 
    bead_params["rvdw_aromatic"] = 4.1 / 2.0 #AutoM3 change: was 4.3 / 2.0    #sigma for ring
    bead_params["rvdw_cross"] = 0.5 * ((4.7 / 2.0) + (4.3 / 2.0))
    bead_params["offset_bd_weight"] =20.0 #AutoM3 change: was 50.0    #penalty weight for nonring beads
    bead_params["offset_bd_aromatic_weight"] = 5.0 #AutoM3 change: was 20.0    #penalty weight for ring beads
    bead_params["lonely_atom_penalize"] = 0.28  #AutoM3 change: was 0.20
    bead_params["bd_bd_overlap_coeff"] = 1.0 #AutoM3 change: was 9.0
    bead_params["at_in_bd_coeff"] = 0.9
    return bead_params


def gaussian_overlap(conformer, bead1, bead2, ringatoms):
    """ "Returns overlap coefficient between two gaussians
    given distance dist"""
    logger.debug("Entering gaussian_overlap()")
    dist = Chem.rdMolTransforms.GetBondLength(conformer, int(bead1), int(bead2))
    bead_params = read_bead_params()
    sigma = bead_params["rvdw"]
    if bead1 in ringatoms and bead2 in ringatoms:
        sigma = bead_params["rvdw_aromatic"]
    if (
        bead1 in ringatoms
        and bead2 not in ringatoms
        or bead1 not in ringatoms
        and bead2 in ringatoms
    ):
        sigma = bead_params["rvdw_cross"]
    return bead_params["bd_bd_overlap_coeff"] * math.exp(-(dist**2) / 4.0 / sigma**2)


def atoms_in_gaussian(molecule, conformer, bead_id, ringatoms):
    """Returns weighted sum of atoms contained in bead bead_id"""
    logger.debug("Entering atoms_in_gaussian()")
    weight_sum = 0.0
    bead_params = read_bead_params()
    sigma = bead_params["rvdw"]
    lumped_atoms = []
    if bead_id in ringatoms:
        sigma = bead_params["rvdw_aromatic"]
    for i in range(conformer.GetNumAtoms()):
        dist_bd_at = Chem.rdMolTransforms.GetBondLength(conformer, i, int(bead_id))
        if dist_bd_at < sigma:
            lumped_atoms.append(i)
        weight_sum -= molecule.GetAtomWithIdx(i).GetMass() * math.exp(
            -(dist_bd_at**2) / 2 / sigma**2
        )
    return bead_params["at_in_bd_coeff"] * weight_sum, lumped_atoms


def penalize_lonely_atoms(molecule, conformer, lumped_atoms):
    """Penalizes configuration if atoms aren't included
    in any CG bead"""
    logger.debug("Entering penalize_lonely_atoms()")
    weight_sum = 0.0
    bead_params = read_bead_params()
    num_atoms = conformer.GetNumAtoms()
    atoms_array = np.arange(num_atoms)
    for i in np.nditer(np.arange(atoms_array.size)):
        if atoms_array[i] not in lumped_atoms:
            weight_sum += molecule.GetAtomWithIdx(int(atoms_array[i])).GetMass()
    return bead_params["lonely_atom_penalize"] * weight_sum


def eval_gaussian_interac(molecule, conformer, list_beads, ringatoms):
    """From collection of CG beads placed on mol, evaluate
    objective function of interacting beads"""
    logger.debug("Entering eval_gaussian_interac()")

    weight_sum = 0.0
    weight_overlap = 0.0
    weight_at_in_bd = 0.0
    bead_params = read_bead_params()

    # Offset energy for every new CG bead.
    # Distinguish between aromatics and others.
    num_aromatics = 0
    lumped_atoms = []

    # Creat list_beads array and loop over indices
    list_beads_array = np.asarray(list_beads)
    for i in np.nditer(np.arange(list_beads_array.size)):
        if list_beads_array[i] in ringatoms:
            num_aromatics += 1
    weight_offset_bd_weights = (
        bead_params["offset_bd_weight"] * (list_beads_array.size - num_aromatics)
        + bead_params["offset_bd_aromatic_weight"] * num_aromatics
    )
    weight_sum += weight_offset_bd_weights

    # Repulsive overlap between CG beads
    for i in np.nditer(np.arange(list_beads_array.size)):
        if i < list_beads_array.size - 1:
            for j in np.nditer(np.arange(i + 1, list_beads_array.size)):
                weight_overlap += gaussian_overlap(
                    conformer, list_beads_array[i], list_beads_array[j], ringatoms
                )
    weight_sum += weight_overlap

    # Attraction between atoms nearby to CG bead
    for i in np.nditer(np.arange(list_beads_array.size)):
        weight, lumped = atoms_in_gaussian(molecule, conformer, list_beads_array[i], ringatoms)
        weight_at_in_bd += weight
        lumped_array = np.asarray(lumped)
        for j in np.nditer(np.arange(lumped_array.size)):
            if lumped_array[j] not in lumped_atoms:
                lumped_atoms.append(lumped_array[j])
    weight_sum += weight_at_in_bd
    # Penalty for excluding atoms
    weight_lonely_atoms = penalize_lonely_atoms(molecule, conformer, lumped_atoms)
    weight_sum += weight_lonely_atoms
    return weight_sum


def check_beads(molecule, list_heavyatoms, heavyatom_coords, trial_comb, ring_atoms, listbonds):
    """Check if CG bead positions in trailComb are acceptable"""
    logger.debug("Entering check_beads()")
    acceptable_trial = ""
    # Check for beads at the same place
    count = Counter(trial_comb)
    all_different = True
    for val in count.values():
        if val != 1:
            all_different = False
            acceptable_trial = False
            logger.debug("Error. Multiple beads on the same atom position for %s" % trial_comb)
            break
    if all_different:
        acceptable_trial = True
        # Check for beads linked by chemical bond (except in rings)
        bonds_in_rings = [0] * len(ring_atoms)
        for bi in range(len(trial_comb)):
            for bj in range(bi + 1, len(trial_comb)):
                if [trial_comb[bi], trial_comb[bj]] in listbonds or [
                    trial_comb[bj],
                    trial_comb[bi],
                ] in listbonds:
                    bond_in_ring = False
                    for r in range(len(ring_atoms)):
                        if trial_comb[bi] in ring_atoms[r] and trial_comb[bj] in ring_atoms[r]:
                            bonds_in_rings[r] += 1
                            bond_in_ring = True
                    if not bond_in_ring:
                        acceptable_trial = False
                        logger.debug("Error. No bond in ring for %s" % trial_comb)
                        break
        if acceptable_trial:
            # Don't allow bonds between atoms of the same ring.
            for bir in range(len(bonds_in_rings)):
                if bonds_in_rings[bir] > 0:
                    logger.debug("Error. Bonds between atoms of the same ring for %s" % trial_comb)
                    acceptable_trial = False
        if acceptable_trial:
            # Check for two terminal beads linked by only one atom
            for bi in range(len(trial_comb)):
                for bj in range(bi + 1, len(trial_comb)):
                    if (
                        [item for sublist in listbonds for item in sublist].count(trial_comb[bi])
                        == 1
                    ) and (
                        [item for sublist in listbonds for item in sublist].count(trial_comb[bj])
                        == 1
                    ):
                        # Both beads are on terminal atoms. Block contribution
                        # if the two terminal atoms are linked to the same atom.
                        partneri = ""
                        partnerj = ""
                        for bond in listbonds:
                            if bond[0] == trial_comb[bi]:
                                partneri = bond[1]
                            if bond[1] == trial_comb[bi]:
                                partneri = bond[0]
                            if bond[0] == trial_comb[bj]:
                                partnerj = bond[1]
                            if bond[1] == trial_comb[bj]:
                                partnerj = bond[0]
                        if partneri == partnerj:
                            acceptable_trial = False
                            logger.debug(
                                "Error. Two terminal beads linked to the same atom for %s"
                                % trial_comb
                            )
    return acceptable_trial

### AutoM3 change :  Including Ertl Functional Groups Finder algorithm (merge, identify_functional_groups) ###

def merge(mol, marked, aset): # AutoM3 change
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


def identify_functional_groups(mol): # AutoM3 change
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
    """for ix, fg in enumerate(ifgs):
        print(f'Functional Group {ix + 1}:')
        print(f'  Atom Indices: {fg.atomIds}')
        print(f'  Atoms (SMILES): {fg.atoms}')
        print(f'  Group Type (SMILES): {fg.type}')
        print(f'  Group Type Atom Indices: {fg.type_atomIds}')"""
    
    """
    USE:
    m = Chem.MolFromSmiles(smiles)
    fgs = identify_functional_groups(m)
    print('%2d: %d fgs' % (ix + 1, len(fgs)), fgs)
    """
    return ifgs

def identify_functional_groups_original(mol):
    # atoms connected by non-aromatic double or triple bond to any heteroatom
    # c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
    PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
    # atoms in non aromatic carbon-carbon double or triple bonds
    PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
    # acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
    PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
    # all atoms in oxirane, aziridine and thiirane rings
    PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

    PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)

    marked = set()
    #mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    #mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    #merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)


    #extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(
        ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
            type=Chem.MolFragmentToSmiles(mol, g.union(uca), canonical=True)))
    """
    USE:
    m = Chem.MolFromSmiles(smiles)
    fgs = identify_functional_groups(m)
    print('%2d: %d fgs' % (ix + 1, len(fgs)), fgs)
    """
    return ifgs

def find_bead_pos(
    molecule, conformer, list_heavy_atoms, heavyatom_coords, ring_atoms, ringatoms_flat, force_map
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
    # List of bonds between heavy atoms
    list_bonds = []
    for i in range(len(list_heavy_atoms)):
        for j in range(i + 1, len(list_heavy_atoms)):
            if (
                molecule.GetBondBetweenAtoms(int(list_heavy_atoms[i]), int(list_heavy_atoms[j]))
                is not None
            ):
                list_bonds.append([list_heavy_atoms[i], list_heavy_atoms[j]])

    ### AutoM3 change : Max and Min number of beads --> in Martini3 it can be 2 to 4 heavy atoms per bead ###
    max_beads = int(len(list_heavy_atoms) / 2.0)
    min_beads = int(len(list_heavy_atoms) / 4.0)

    # Collect all possible combinations of bead positions
    best_trial_comb = []
    list_trial_comb = []
    ene_best_trial = 1e6
    last_best_trial_comb = []

    # Keep track of all combinations and scores
    list_combs = []
    list_energies = []

    for num_beads in range(min_beads,max_beads+1):

        # Use recursive function to loop through all possible
        # combinations of CG bead positions.
        if num_beads==0: num_beads=1
        seq_one_beads = np.array(list(itertools.combinations(list_heavy_atoms, num_beads)))
        combs = []
        energies = []

        # Trial positions: any heavy atom
        for seq in seq_one_beads:
            trial_comb = list(seq)
            acceptable_trial = check_beads(
                molecule, list_heavy_atoms, heavyatom_coords, trial_comb, ring_atoms, list_bonds
            ) # AutoM3 change : Added molecule argument

            if acceptable_trial:

                # Do the energy evaluation
                trial_ene = eval_gaussian_interac(molecule, conformer, trial_comb, ringatoms_flat)
                combs.append(trial_comb)
                energies.append(trial_ene)

                logger.info("; %s %s", trial_comb, trial_ene)
                # Make sure all atoms within one bead would be connected
                if all_atoms_in_beads_connected(
                    trial_comb, heavyatom_coords, list_heavy_atoms, list_bonds, molecule, force_map
                ): # AutoM3 change : Added molecule and force_map arguments
                    
                    # Accept the move
                    if trial_ene < ene_best_trial:
                        ene_best_trial = trial_ene
                        best_trial_comb = sorted(trial_comb)
                    # Get bead positions
                    beadpos = [[0] * 3 for l in range(len(trial_comb))]
                    for l in range(len(trial_comb)):
                        beadpos[l] = [
                            conformer.GetAtomPosition(int(sorted(trial_comb)[l]))[m]
                            for m in range(3)
                        ]
                    # Store configuration
                    list_trial_comb.append([trial_comb, beadpos, trial_ene])

        if last_best_trial_comb == best_trial_comb:
            break

        last_best_trial_comb = best_trial_comb
        list_combs.append(combs)
        list_energies.append(energies)

    sorted_combs = np.array(sorted(list_trial_comb, key=itemgetter(2)), dtype="object")
    return sorted_combs[:, 0], sorted_combs[:, 1]

def all_atoms_in_beads_connected(trial_comb, heavyatom_coords, list_heavyatoms, bondlist, mol,force_map): #AutoM3 change: added mol, force_map
    """Make sure all atoms within one CG bead are connected to at least
    one other atom in that bead"""
    logger.debug("Entering all_atoms_in_beads_connected()")
    # Bead coordinates are given by heavy atoms themselves
    cgbead_coords = []

    for i in range(len(trial_comb)):
        cgbead_coords.append(heavyatom_coords[list_heavyatoms.index(trial_comb[i])])
    
    _, num_arom = topology.is_aromatic(mol) #AutoM3 change

    ### AutoM3 change of mapping approach to differenciate molecules with 0-1 and more cycles
    if not force_map and num_arom<7: #AutoM3 change
        voronoi, _  = voronoi_atoms_new(cgbead_coords, heavyatom_coords) #AutoM3 change
    else:
        voronoi, _  = voronoi_atoms_old(cgbead_coords, heavyatom_coords) #AutoM3 change
    logger.debug("voronoi %s" % voronoi)

    for i in range(len(trial_comb)):
        cg_bead = trial_comb[i]
        num_atoms = list(voronoi.values()).count(voronoi[list_heavyatoms.index(cg_bead)])
        # sub-part of bond list that only contains atoms within CG bead
        sub_bond_list = []
        for j in range(len(bondlist)):
            if (
                voronoi[list_heavyatoms.index(bondlist[j][0])]
                == voronoi[list_heavyatoms.index(cg_bead)]
                and voronoi[list_heavyatoms.index(bondlist[j][1])]
                == voronoi[list_heavyatoms.index(cg_bead)]
            ):
                sub_bond_list.append(bondlist[j])
        num_bonds = len(sub_bond_list)
        if num_bonds < num_atoms - 1 or num_atoms == 1:
            logger.debug("Error: Not all atoms in beads connected in %s" % trial_comb)
            logger.debug("Error: %s < %s, %s" % (num_bonds, num_atoms - 1, sub_bond_list))
            return False
    return True

def voronoi_atoms_new(cgbead_coords, heavyatom_coords): # AutoM3
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
                if (cgbead_coords[i] != heavyatom_coords[j]).all():
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
        for i in partitioning.values():
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
            partitioning[j] = len(cgbead_coords)
    
    # initiating dict with beads as keys and their coordinates as list of values
    bead_coord={}
    for bead in partitioning.values():
        bead_coord[bead]=[]

    for atom in range(len(heavyatom_coords)):
        bead=partitioning[atom]
        bead_coord[bead].append(heavyatom_coords[atom])
    
    bead_cog=[]
    for bead, coords in bead_coord.items():
        bead_coord = np.stack(coords, axis=0)
        cog = np.mean(bead_coord, axis=0)
        bead_cog.append(cog)
    return partitioning, bead_cog

def voronoi_atoms_old(cgbead_coords, heavyatom_coords): #AutoM3 change
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
        for i in partitioning.values():
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

    # AutoM3 change : finding the bead center coordinate for COG 
    bead_coord={}
    for bead in partitioning.values():
        bead_coord[bead]=[]

    for atom in range(len(heavyatom_coords)):
        bead=partitioning[atom]
        bead_coord[bead].append(heavyatom_coords[atom])
    
    bead_cog=[]
    for bead, coords in bead_coord.items():
        bead_coord = np.stack(coords, axis=0)
        cog = np.mean(bead_coord, axis=0)
        bead_cog.append(cog)
    return partitioning, bead_cog

def functional_groups_ok(atom_partitioning,molecule,ringatoms): # AutoM3
    """
    Checking if functional groups are conserved in distinctive bead, within atom number per bead limit.
    """

    fgs = identify_functional_groups(molecule)
    fgs_id=[j[0] for j in [i[0] for i in fgs]]

    bead_atoms={}
    for at, bead in atom_partitioning.items():
        if bead not in bead_atoms:
            bead_atoms[bead] = []
        bead_atoms[bead].append(at) 
    
    group_found = []
    for ix, fg in enumerate(fgs):
        gr_f = False

        """if fg.atoms=='cc': #bridge atoms between fused cycles
            for bead, atoms in bead_atoms.items():
                bridge_lost=False
                if not set(fg.atomIds).issubset(atoms):
                    print(f"the found bridge {fg.atomIds} is not in  {atoms}")
                    bridge_lost=True
            if bridge_lost: return False # reject mapping which doesn't enclose bridge in one bead"""
        
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

def max2arperbead(atom_partitioning, ringatoms): # AutoM3
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
        else:   
            return True
