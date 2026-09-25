"""GROMACS .itp topology text generation.

Turns a resolved CG-bead mapping (atoms, bonds, angles, dihedrals, optional
virtual sites) into GROMACS .itp text, plus the auxiliary Bartender input
format used to fit bonded parameters.
"""

from ..common import *
from .molecule import gen_molecule_smi, get_charge
from .beadtypes import (
    substruct2smi,
    letter_occurrences,
    determine_bead_type,
    get_standard_mass,
    read_params,
)
from .alogps import smi2alogps

logger = logging.getLogger(__name__)


def print_header(molname, mol_smi):
    """Print topology header"""
    text = "; GENERATED WITH Auto_Martini M3FF for {}\n".format(molname)

    info = (
        "; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour\n"
        + "; updated to Martini 3 force field by Magdalena Szczuka\n"
        + "; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza \n"
        + "; SMILES code : "+mol_smi +"\n\n"
        + "\n[moleculetype]\n"
        + "; molname       nrexcl\n"
        + "  {:5s}         2\n\n".format(molname)
        + "[atoms]\n"
        + "; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num"
    )
    return text + info


def print_atoms(molname,forcepred,cgbeads,molecule,hbonda,hbondd,partitioning,ringatoms,ringatoms_flat,logp_file,trial=False):
    """Print CG Atoms in itp format"""

    logger.debug("Entering print_atoms()")
    atomnames = []
    beadtypes = []
    text = ""
    atoms_in_smi_dict={}

    for bead in range(len(cgbeads)):
        # Determine SMI of substructure
        try:
            smi_frag, wc_log_p, charge, atoms_in_smi, converted_smi, real_smi  = substruct2smi(
                molecule, partitioning, bead
            )
        except Exception:
            raise
        atoms_in_smi_dict[bead+1]=atoms_in_smi.replace(" ; atoms: ","")

        atom_name = ""
        for character, count in sorted(six.iteritems(letter_occurrences(smi_frag))):
            try:
                float(character)
            except ValueError:
                if count == 1:
                    atom_name += "{:s}".format(character)
                else:
                    atom_name += "{:s}{:s}".format(character, str(count))

        # Get charge for smi_frag
        mol_frag, errval = gen_molecule_smi(smi_frag)
        charge_frag = get_charge(mol_frag)

        if errval == 0:
            charge_frag = get_charge(mol_frag)

            # Extract ALOGPS free energy
            # logporigin records whether the value came from ALOGPS, so it can
            # be noted in the itp output's comment column.
            try:
                if charge_frag == 0:
                    alogps, logporigin = smi2alogps(forcepred, smi_frag, wc_log_p, bead + 1,converted_smi, real_smi,logp_file, trial)
                else:
                    alogps = 0.0
            except (NameError, TypeError, ValueError):
                return atomnames, beadtypes, errval

            hbond_a_flag = 0
            for at in hbonda:
                if partitioning[at] == bead:
                    hbond_a_flag = 1
                    break
            hbond_d_flag = 0
            for at in hbondd:
                if partitioning[at] == bead:
                    hbond_d_flag = 1
                    break

            in_ring = cgbeads[bead] in ringatoms_flat

            bead_type = determine_bead_type(alogps, charge, hbond_a_flag, hbond_d_flag, in_ring, smi_frag)
            atom_name = ""
            name_index = 0
            while atom_name in atomnames or name_index == 0:
                name_index += 1
                atom_name = "{:1s}{:02d}".format(bead_type[1], name_index)
            atomnames.append(atom_name)

            mass = get_standard_mass(bead_type)

            if not trial:
                if len(molname)>4:molname=molname[:4]
                text = (
                    text
                    + "   {:<5d}   {:5s}   1   {:5s}   {:7s}   {:<5d}   {:2d}   {:3d}   ;   {:8s}{:8s}{:9s}\n".format(
                        bead + 1,
                        bead_type,
                        molname,
                        atom_name,
                        bead + 1,
                        charge,
                        mass,
                        smi_frag,
                        atoms_in_smi,
                        logporigin
                    )
                )
            beadtypes.append(bead_type)

    return atomnames, beadtypes, text, atoms_in_smi_dict


def print_bonds(cgbeads, cgbeads_ring, molecule, partitioning, cgbead_coords, beadtypes, ringatoms, trial=False):
    """print CG bonds in itp format"""

    logger.debug("Entering print_bonds()")

    # Bond information
    bondlist = []
    constlist = []
    text = ""
    cpt_ringatoms = 0

    if ringatoms != []:
        cpt_ringatoms=len(sum(ringatoms,[]))
    if len(cgbeads) > 1:
        for i in range(len(cgbeads)):
            for j in range(i + 1, len(cgbeads)):
                dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                if dist < 0.61:
                    # Are atoms part of the same ring
                    for ring in ringatoms:
                        if cgbeads[i] in ring and cgbeads[j] in ring and [i, j, dist] not in constlist:
                            constlist.append([i, j, dist])

                    if dist < 0.134:
                        raise NameError("Bond too short")

                    # Look for a bond between an atom of i and an atom of j
                    found_connection = False
                    atoms_in_bead_i = []
                    for ii in partitioning.keys():
                        if partitioning[ii] == i:
                            atoms_in_bead_i.append(ii)

                    atoms_in_bead_j = []
                    for jj in partitioning.keys():
                        if partitioning[jj] == j:
                            atoms_in_bead_j.append(jj)
                    for ib in range(len(molecule.GetBonds())):
                        abond = molecule.GetBondWithIdx(ib)
                        if (
                            abond.GetBeginAtomIdx() in atoms_in_bead_i
                            and abond.GetEndAtomIdx() in atoms_in_bead_j
                        ) or (
                            abond.GetBeginAtomIdx() in atoms_in_bead_j
                            and abond.GetEndAtomIdx() in atoms_in_bead_i
                        ):
                            found_connection = True

                    if found_connection:
                        bondlist.append([i, j, dist])

                    else:
                        if cpt_ringatoms<7 and len(cgbeads)<5 and [i, j, dist] not in constlist:
                            constlist.append([i, j, dist])

        # Check if there are beads with ring atoms that are not connected
        for ir in range(len(cgbeads_ring)):
            for jr in range(ir + 1, len(cgbeads_ring)):
                distr = np.linalg.norm(cgbead_coords[ir] - cgbead_coords[jr]) * 0.1
                if distr < 0.65:
                    # Are atoms part of the same ring
                    for ring in ringatoms:
                        if ( cgbeads_ring[ir] in ring and cgbeads_ring[jr] in ring and distr<=0.45
                            ) and ([ir, jr, distr] not in constlist and [ir, jr, distr] not in  bondlist ):
                            constlist.append([ir, jr, distr])

        # Go through list of constraints. If we find an extra
        # possible constraint between beads that have constraints,
        # add it.
        beads_with_const = []
        for c in constlist:
            if c[0] not in beads_with_const:
                beads_with_const.append(c[0])
            if c[1] not in beads_with_const:
                beads_with_const.append(c[1])

        beads_with_const = sorted(beads_with_const)
        for i in range(len(beads_with_const)):
            for j in range(1 + i, len(beads_with_const)):
                const_exists = False
                for c in constlist:
                    if (c[0] == i and c[1] == j) or (c[0] == j and c[1] == i):
                        const_exists = True
                        break
                if not const_exists:
                    dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                    if any(dist  != bl[2] for bl in bondlist):
                        # Check that it's not in the bond list
                        in_bond_list = False
                        for b in bondlist:
                            if (b[0] == i and b[1] == j) or (b[0] == j and b[0] == i):
                                in_bond_list = True
                                break
                        # Are atoms part of the same ring
                        in_ring = False
                        for ring in ringatoms:
                            if cgbeads[i] in ring and cgbeads[j] in ring and len(ring)<5:
                                in_ring = True
                                break
                        # If not in bondlist and in the same ring, add the contraint
                        if not in_bond_list and in_ring and [i, j, dist] not in constlist:
                            constlist.append([i, j, dist])

        if not trial:
            beadlist=[]
            for bead in beadtypes:
                if not bead.startswith('T') and not bead.startswith('S'): beadlist.append('R')
                else: beadlist.append(bead[0])

            if len(bondlist) > 0:
                text = "\n[bonds]\n" + ";  i   j     funct   length   force.c."
                for b in bondlist:
                    # Make sure atoms in bond are not part of the same ring
                    text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}       {:4.2f}".format(
                        b[0] + 1, b[1] + 1, b[2], read_params(b[2],beadlist[b[0]]+"-"+beadlist[b[1]])
                    )
            else: text = "\n[bonds]\n"

            if len(constlist) > 0:
                text = text + "\n[constraints]\n" + ";  i   j     funct   length"

                for c in constlist:
                    if c not in bondlist:
                        if cpt_ringatoms>18 and c[2]>0.415:
                            continue
                        text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}".format(
                            c[0] + 1, c[1] + 1, c[2]
                        )
            # Make sure there's at least a bond to every atom
            for i in range(len(cgbeads)):
                bond_to_i = False
                for b in bondlist + constlist:
                    if i in [b[0], b[1]]:
                        bond_to_i = True
                if not bond_to_i:
                    print("Error. No bond to atom %d" % (i + 1))
                    exit(1)
    return bondlist, constlist, text


def print_angles(cgbeads, molecule, partitioning, cgbead_coords, beadtypes, bondlist, constlist, ringatoms):
    """print CG angles in itp format and returns the angles list"""
    logger.debug("Entering print_angles()")

    text = ""
    angle_list = []

    if len(cgbeads) > 2:
        # Angles
        for i in range(len(cgbeads)):
            for j in range(len(cgbeads)):
                for k in range(len(cgbeads)):
                    all_in_ring = False
                    for ring in ringatoms:
                        if cgbeads[i] in ring and cgbeads[j] in ring and cgbeads[k] in ring:
                            all_in_ring = True
                            break
                    # Forbid all atoms linked by constraints
                    all_constraints = False
                    ij_bonded = False
                    jk_bonded = False
                    ij_const = False
                    jk_const = False
                    for b in bondlist + constlist:
                        if i in [b[0], b[1]] and j in [b[0], b[1]]:
                            ij_bonded = True
                            if b in constlist:
                                ij_const = True
                        if j in [b[0], b[1]] and k in [b[0], b[1]]:
                            jk_bonded = True
                            if b in constlist:
                                jk_const = True
                    if ij_const and jk_const:
                        all_constraints = True
                    if (
                        not all_in_ring
                        and (ij_bonded or jk_bonded)
                        and i != j
                        and j != k
                        and i != k
                        and not all_constraints
                    ):
                        # Measure angle between i, j, and k.
                        angle = (
                            180.0
                            / math.pi
                            * math.acos(
                                np.dot(
                                    cgbead_coords[i] - cgbead_coords[j],
                                    cgbead_coords[k] - cgbead_coords[j],
                                )
                                / (
                                    np.linalg.norm(cgbead_coords[i] - cgbead_coords[j])
                                    * np.linalg.norm(cgbead_coords[k] - cgbead_coords[j])
                                )
                            )
                        )
                        # Look for any double bond between atoms belonging to these CG beads.
                        atoms_in_fragment = []
                        for aa in partitioning.keys():
                            if partitioning[aa] == j:
                                atoms_in_fragment.append(aa)
                        forc_const = 100.0
                        for ib in range(len(molecule.GetBonds())):
                            abond = molecule.GetBondWithIdx(ib)
                            if (
                                abond.GetBeginAtomIdx() in atoms_in_fragment
                                and abond.GetEndAtomIdx() in atoms_in_fragment
                            ):
                                bondtype = molecule.GetBondBetweenAtoms(
                                    abond.GetBeginAtomIdx(), abond.GetEndAtomIdx()
                                ).GetBondType()
                                if bondtype == rdchem.BondType.DOUBLE:
                                    forc_const = 45.0
                        new_angle = True
                        for a in angle_list:
                            if i in a and j in a and k in a:
                                new_angle = False

                        if len(partitioning)>15:
                            for a1 in range(len(angle_list)):
                                for a2 in range(len(angle_list)):
                                    if i in angle_list[a1] and j in angle_list[a1] and j in angle_list[a2] and k in angle_list[a2]:
                                        new_angle = False
                        if new_angle :
                            angle_list.append([i, j, k, angle, forc_const])


        beadlist=[]
        for bead in beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'): beadlist.append('R')
            else: beadlist.append(bead[0])

        if len(angle_list) > 0:
            text = text + "\n[angles]\n"
            text = text + ";  i  j  k    funct  angle  force.c.\n"
            for a in angle_list:
                force = read_params(a[3],beadlist[a[0]]+"-"+beadlist[a[1]]+"-"+beadlist[a[2]])
                if force is None : force=a[4]
                text = text + "  {:2} {:2} {:2}       1    {:<5.1f}  {:5.1f}\n".format(
                    a[0] + 1, a[1] + 1, a[2] + 1, a[3], force
                )
            text = text
    return text, angle_list


def print_dihedrals(cgbeads, constlist, ringatoms, cgbead_coords, beadtypes):
    """Print CG dihedrals in itp format"""
    logger.debug("Entering print_dihedrals()")

    new_dihed_list = []
    text = ""
    num_ar=0

    if len(cgbeads) > 3:
        # Dihedrals
        dihed_list = []
        # Three ring atoms and one non ring
        for i in range(len(cgbeads)):
            for j in range(len(cgbeads)):
                for k in range(len(cgbeads)):
                    for l in range(len(cgbeads)):
                        if i != j and i != k and i != l and j != k and j != l and k != l:

                            three_in_ring = False
                            for ring in ringatoms:
                                num_ar+=len(ring)
                                if [
                                    [cgbeads[i] in ring],
                                    [cgbeads[j] in ring],
                                    [cgbeads[k] in ring],
                                    [cgbeads[l] in ring],
                                ].count([True]) >= 3:
                                    three_in_ring = True
                                    break
                            for b in constlist:
                                if i in [b[0], b[1]] and j in [b[0], b[1]]:
                                    pass
                                if j in [b[0], b[1]] and k in [b[0], b[1]]:
                                    pass
                                if k in [b[0], b[1]] and l in [b[0], b[1]]:
                                    pass
                            # Distance criterion--beads can't be far apart
                            disthres = 0.5
                            close_enough = False
                            if (
                                np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[j] - cgbead_coords[k]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[k] - cgbead_coords[l]) * 0.1
                                < disthres
                            ):
                                close_enough = True

                            already_dih = False
                            for dih in dihed_list:
                                if dih[0] == l and dih[1] == k and dih[2] == j and dih[3] == i:
                                    already_dih = True
                                    break
                            if three_in_ring and close_enough and not already_dih:
                                r1 = cgbead_coords[j] - cgbead_coords[i]
                                r2 = cgbead_coords[k] - cgbead_coords[j]
                                r3 = cgbead_coords[l] - cgbead_coords[k]
                                p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                                p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
                                r2 /= np.linalg.norm(r2)
                                cosphi = np.dot(p1, p2)
                                sinphi = np.dot(r2, np.cross(p1, p2))
                                angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
                                r1_1 = cgbead_coords[i] - cgbead_coords[j]
                                r2_2 = cgbead_coords[j] - cgbead_coords[k]
                                angle_ijk = 180.0 / math.pi * math.acos(np.dot(r1_1,r2) / (np.linalg.norm(r1_1) * np.linalg.norm(r2)))
                                angle_jkl = 180.0 / math.pi * math.acos(np.dot(r2_2,r3) / (np.linalg.norm(r2_2) * np.linalg.norm(r3)))
                                forc_const = 10.0
                                if angle_ijk<145.0 and angle_jkl<145.0 : # look Restricted bending potential in gromacs manual
                                    dihed_list.append([i, j, k, l, angle, forc_const])

        bead_in_ring_coords={}
        for nb,bead_nb in enumerate(cgbeads):
            for ring in ringatoms:
                if bead_nb in ring: bead_in_ring_coords[nb]=cgbead_coords[nb]
        beadlist=[]
        for bead in beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'): beadlist.append('R')
            else: beadlist.append(bead[0])

        new_dihed_list=dihed_list
        if len(dihed_list) > 0:
            text = text + "\n[dihedrals]\n"
            text = text + ";  i  j  k  l  funct  angle  force.c.\n"

            for dl in dihed_list:
                for di in dihed_list[1:]:
                    if dl!=di:
                        # Check if beads are repeating
                        if  dl[0:2]==di[0:2] or dl[0:2]==di[2:4] or dl[2:4]==di[0:2] or dl[2:4]==di[2:4] or sorted(dl[:4])==sorted(di[:4]) :
                            new_dihed_list.remove(di)

            for d in new_dihed_list:
                force=read_params(d[4],beadlist[d[0]]+"-"+beadlist[d[1]]+"-"+beadlist[d[2]]+"-"+beadlist[d[3]])
                if num_ar>0 and (d[0] or d[1] or d[2] or d[3] not in bead_in_ring_coords.keys()) and force is not None: force=force/2 #for dihedral between cycle-bead and non-cycled bead: dicrease of force
                if force is None: force=d[5]
                text = (
                    text
                    + "  {:2} {:2} {:2} {:2}    2    {:<5.1f}  {:5.1f}\n".format(
                        d[0] + 1, d[1] + 1, d[2] + 1, d[3] + 1, d[4], force
                    )
                )
    return text


def print_virtualsites(ringatoms,cg_bead_coords,partitionning,mol):
    """Prints CG virtual sites in itp format."""
    logger.debug("Entering print_virtualsites()")

    text = ""

    #Get number of bonds for each atom
    atom_bond_counts = {atom.GetIdx(): 0 for atom in mol.GetAtoms()}

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        if (mol.GetAtomWithIdx(begin_atom_idx).GetSymbol() != "H" and mol.GetAtomWithIdx(end_atom_idx).GetSymbol() != "H") and (partitionning[begin_atom_idx]!=partitionning[end_atom_idx]): # Check if not hydrogen and not in the same bead
            if begin_atom_idx not in atom_bond_counts:
                atom_bond_counts[begin_atom_idx] = 1
            else : atom_bond_counts[begin_atom_idx] += 1

            if end_atom_idx not in atom_bond_counts:
                atom_bond_counts[end_atom_idx] = 1
            else: atom_bond_counts[end_atom_idx] += 1

    bead_bond_counts = {}
    for a, b in partitionning.items():
        if b not in bead_bond_counts:
            bead_bond_counts[b] = 0
        for at, cpt in atom_bond_counts.items():
            if at == a:
                bead_bond_counts[b] += cpt

    ring_atoms=[]
    virtual_sites={}
    for ra in ringatoms: ring_atoms+=ra

    #Find beads constructing rings
    bead_in_ring_coords={}
    vs_bead_coords=[]

    for atom,bead in partitionning.items():
        if atom in ring_atoms and bead not in bead_in_ring_coords:
            bead_in_ring_coords[bead]=cg_bead_coords[bead]

    #Count distances between each pair of beads
    distances = {}
    for bead, coord in bead_in_ring_coords.items():
        distances[bead]={}
        for other_bead, other_coord in bead_in_ring_coords.items():
            if bead != other_bead:
                distance = np.linalg.norm(coord - other_coord)
                distances[bead][other_bead]=distance

    def find_more_vs(num_vs,bead_bond_counts_sorted,cg_bead_coords,distances):
        vs_bead_coords=[]
        virtual_sites={}
        vs_list=[]
        for i in range(num_vs):
            vs_bead=int(list(bead_bond_counts_sorted.keys())[i])
            vs_list.append(vs_bead)

            for j in range(len(cg_bead_coords)):
                if j==vs_bead: vs_bead_coords.append(cg_bead_coords[i])

        for vs in vs_list:
            constructing_beads_dist=dict(sorted(distances[vs].items(), key=lambda item: item[1]))
            constructing_beads=[bead for bead in constructing_beads_dist.keys()]
            for bead in constructing_beads:
                if bead in vs_list: constructing_beads.remove(bead)

            if vs not in virtual_sites.keys():
                virtual_sites[vs]=constructing_beads[:4]
        return virtual_sites


    #Find number of fused cycles = number of needed virtual sites
    bead_bond_counts_sorted = dict(sorted(bead_bond_counts.items(), key=lambda item: item[1], reverse=True))
    cpt_ringatoms=len(sum(ringatoms,[]))

    for r_nb in range(len(ringatoms)):
        if cpt_ringatoms>6 and cpt_ringatoms<19 :
            virtual_sites=find_more_vs(1,bead_bond_counts_sorted,cg_bead_coords,distances)

        if cpt_ringatoms>18: # more than 4 fused cycles
            virtual_sites=find_more_vs(3,bead_bond_counts_sorted,cg_bead_coords,distances)

    text = text + "\n[virtual_sitesn]\n"
    text = text + "; site funct  constructing atom indices"
    rigid_dihedral = []
    for vs, cb in virtual_sites.items():
        if len(cb)==4:
            text = (text + "\n   {:d}       1     {:d} {:d} {:d} {:d}".format(
                                    vs+1, cb[0] + 1, cb[1] + 1, cb[2] + 1, cb[3] + 1
                                )
                            )

            # Find dihedral from constructing beads
            i=cb[0]
            j=cb[1]
            k=cb[2]
            l=cb[3]
            r1 = cg_bead_coords[j] - cg_bead_coords[i]
            r2 = cg_bead_coords[k] - cg_bead_coords[j]
            r3 = cg_bead_coords[l] - cg_bead_coords[k]
            p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
            p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
            r2 /= np.linalg.norm(r2)
            cosphi = np.dot(p1, p2)
            sinphi = np.dot(r2, np.cross(p1, p2))
            angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
            force=100
            new_dih="  {:2} {:2} {:2} {:2}    2    {:<5.1f}  {:5.1f}".format(cb[0]+1,cb[1]+1,cb[2]+1,cb[3]+1, round(angle,2), force)
            rigid_dihedral.append(new_dih)

        if len(cb)==3:
            text = (text + "\n   {:d}       1     {:d} {:d} {:d}".format(
                                    vs+1, cb[0] + 1, cb[1] + 1, cb[2] + 1
                                )
                            )
        if len(cb)==2:
            text = (text + "\n   {:d}       1     {:d} {:d}".format(
                                    vs+1, cb[0] + 1, cb[1] + 1
                                )
                            )

    return text, virtual_sites, rigid_dihedral


def topout(header_write,atoms_write,bonds_write,angles_write):
    """Print simple itp file"""
    text=header_write +"\n"+ atoms_write + "\n" + bonds_write + "\n" + angles_write

    #bartender info search
    bartender_input_info={}
    bartender_input_info["BONDS"]=[]
    for line in list(bonds_write.split("\n")):
        if ";" not in line and len(line.split())>4:
            bartender_input_info["BONDS"].append(line.split()[:2])

    bartender_input_info["ANGLES"]=[]
    for line in list(angles_write.split("\n")):
        if ";" not in line and len(line.split())>5:
            bartender_input_info["ANGLES"].append(line.split()[:3])
    return text, bartender_input_info


def topout_noVS(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, bead_coords, ring_atoms, cg_beads):
    """Print itp file without virtual sites, upon user's wish"""
    text = ""

    molname=""
    for line in list(atoms_write.split("\n")):
        if line != "":
            x = line.split()
            molname=x[3]
    modified_header_write=header_write
    modified_bonds_write=bonds_write
    exclusions_net=""
    if len(ring_atoms[0])>4 and len(ring_atoms[0])<10 and len(bead_coords)<6:
        #changing nrexcl to 1 if 1 cycle and max 5 beads
        modified_lines_header=[]
        for line in list(header_write.split("\n")):
            if ("  "+molname) not in line: modified_lines_header.append(line)
            else:
                lineH=line.split("         ")
                txt=lineH[0]+"          1"
                modified_lines_header.append(txt)
        modified_header_write="\n".join(modified_lines_header)

        #Adding force to constraints
        modified_lines_bonds=[]
        for line in list(bonds_write.split("\n")):
            if "1" in line and len(line.split("   "))<7:
                modified_lines_bonds.append(line+"    1000000")
            else: modified_lines_bonds.append(line)
            if line=="[constraints]":
                if line in modified_lines_bonds : modified_lines_bonds.remove(line)
                txt = "#ifndef FLEXIBLE\n[constraints]\n#endif"
                modified_lines_bonds.append(txt)

        #adding exclusions for two most distant beads in ring
        if len(bead_coords)>3:
            remote_dist=0
            remote_beads = []
            bead_in_ring_coords={}
            ring_atoms=ring_atoms[0]

            for nb,bead_nb in enumerate(cg_beads):
                bead_in_ring_coords[nb+1]=bead_coords[nb]

            for nb_bead1, coord1 in bead_in_ring_coords.items():
                for nb_bead2, coord2 in bead_in_ring_coords.items():
                    dist= math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

                    if dist > remote_dist and nb_bead1!=nb_bead2:
                        remote_beads=[nb_bead1,nb_bead2]
                        remote_dist=dist
            exclusions_net=""
            exclusions_net = exclusions_net + "\n[exclusions]\n"
            exclusions_net = exclusions_net + "  " + str(remote_beads[0])+ " " + str(remote_beads[1])
            exclusions_net=exclusions_net+"\n"

            for line in modified_lines_bonds:
                if line!="" and len(line.split("   "))>6:
                    if str(remote_beads[0]) == line.split("   ")[1] and str(remote_beads[1]) == line.split("   ")[2] :
                        if line in modified_lines_bonds : modified_lines_bonds.remove(line)
                    else:
                        if str(remote_beads[1]) == line.split("   ")[1] and str(remote_beads[0]) == line.split("   ")[2] :
                            if line in modified_lines_bonds : modified_lines_bonds.remove(line)


        modified_bonds_write="\n".join(modified_lines_bonds)

    if len(cg_beads)>4:
        #Clean angles already described by dihedrals
        modified_lines_angles = []
        for lineA in list(angles_write.split("\n")):
            if lineA not in modified_lines_angles: modified_lines_angles.append(lineA)
            for lineD in list(dihedrals_write.split("\n")):
                angle_line = lineA.split()
                dihed_line = lineD.split()
                if len(dihed_line)>2 and not lineD.startswith(";") and len(angle_line)>2 and not lineA.startswith(";"):
                    if angle_line[0] in dihed_line[:4] and angle_line[1] in dihed_line[:4] and angle_line[2] in dihed_line[:4] and lineA in modified_lines_angles:
                        modified_lines_angles.remove(lineA)
        modified_angles_write = "\n".join(modified_lines_angles)
    else : modified_angles_write = angles_write

    #bartender info search
    bartender_input_info={}
    bartender_input_info["BONDS"]=[]
    for line in list(modified_bonds_write.split("\n")):
        if ";" not in line and len(line.split())>4:
            bartender_input_info["BONDS"].append(line.split()[:2])

    bartender_input_info["ANGLES"]=[]
    for line in list(modified_angles_write.split("\n")):
        if ";" not in line and len(line.split())>5:
            bartender_input_info["ANGLES"].append(line.split()[:3])

    bartender_input_info["IMPROPERS"]=[]
    for line in list(dihedrals_write.split("\n")):
        if ";" not in line and len(line.split())>6:
            bartender_input_info["IMPROPERS"].append(line.split()[:4])

    text = modified_header_write +"\n"+ atoms_write +"\n"+ modified_bonds_write +"\n"+ modified_angles_write +"\n"+ dihedrals_write+exclusions_net
    return text, bartender_input_info


def topout_vs(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, virtual_sites, vs_write, rigid_dih, simple_model):
    """Prints whole .itp file with all bonded and nonbonded parameters."""
    text = ""
    bartender_input_info={}
    nb_beads=0
    molname=""
    for line in list(atoms_write.split("\n")):
        if line != "":
            x = line.split()
            molname=x[3]
            nb_beads=int(x[0])

    #Atoms: add bead VS
    vs_bead_names=""
    #Atoms: change mass of VS to 0 and divide it between constructing beads
    modified_lines_atoms = list(atoms_write.split("\n"))
    vs_mass={}
    for vs, cb in virtual_sites.items():
        for i, line in enumerate(modified_lines_atoms):
            if line:
                atom_line = line.split()
                if str(vs+1) == atom_line[0] and atom_line[7] != '0':
                    vs_bead_names+=atom_line[1]
                    vs_mass[vs]=int(atom_line[7])
                    fields = line.split()
                    comments = line.split('   ;   ')[1]
                    modified_lines_atoms[i] = "   {:<5d}   {:5s}   1   {:5s}   {:7s}   {:<5d}   {:2d}     0   ;   {:24s}".format(
                        int(fields[0]), fields[1],  str(fields[3]), fields[4], int(fields[5]), int(fields[6]), comments)

    for vs, cb in virtual_sites.items():
        for vs_env in cb:
            for j, line2 in enumerate(modified_lines_atoms):
                if line2 :
                    atom_line2 = line2.split()
                    if str(vs_env+1) == atom_line2[0]:
                        new_mass = int(int(atom_line2[7]) + vs_mass[vs] / len(cb) ) #add 1/cb mass of VS
                        fields = line2.split()
                        comments = line2.split('   ;   ')[1]
                        modified_lines_atoms[j] = "   {:<5d}   {:5s}   1   {:5s}   {:7s}   {:<5d}   {:2d}   {:3d}   ;   {:24s}".format(
                            int(fields[0]), fields[1],  str(fields[3]), fields[4], int(fields[5]), int(fields[6]),
                            int(new_mass), comments
)
    modified_atoms_write = "\n".join(modified_lines_atoms)

    modified_lines_header=[]
    for line in list(header_write.split("\n")):
        if ("  "+molname) not in line: modified_lines_header.append(line)
        else:
            lineH=line.split("         ")
            txt=lineH[0]+"         1"
            modified_lines_header.append(txt)
    modified_header_write="\n".join(modified_lines_header)

    #Adding force to constraints
    modified_lines_bonds=[]
    bonds_list = []
    for line in list(bonds_write.split("\n")):
        if '1' in line:
            bonds_list.append(f"{line.split('   ')[1]},{line.split('   ')[2]}")
        if '1' in line and len(line.split("   "))<7:
            modified_lines_bonds.append(line+"    1000000")
        else:
            modified_lines_bonds.append(line)
        if line=="[constraints]":
            if line in modified_lines_bonds : modified_lines_bonds.remove(line)
            txt = "#ifndef FLEXIBLE\n[constraints]\n#endif"
            modified_lines_bonds.append(txt)
    modified_bonds_write = "\n".join(modified_lines_bonds)

    #Bonds / Constraints: delete lines describing interactions with VS
    bond_with_vs={}
    for line in list(modified_bonds_write.split("\n")):
        if line !="":
            bond_line = line.split()
            if len(bond_line)>2 and not line.startswith(";"):
                for vs, cb in virtual_sites.items():
                    if str(vs+1) in bond_line[:2]:
                        # memorizing atom bonded with VS = vs_bond
                        bonded_to_vs = bond_line[1] if str(vs+1) == bond_line[0] else bond_line[0]

                        #memorize VS bond count
                        if str(vs+1) not in bond_with_vs:
                            bond_with_vs[str(vs+1)]=[]
                        if bonded_to_vs not in bond_with_vs[str(vs+1)]:
                            bond_with_vs[str(vs+1)].append(bonded_to_vs) #beads bounded to VS

                        #Check if bond between bead B and VS is the only bond connecting B to the rest of the molecule: if yes, don't remove it
                        if line in modified_lines_bonds and len(bond_with_vs[str(vs+1)])>1:
                            modified_lines_bonds.remove(line)
    modified_bonds_write = "\n".join(modified_lines_bonds)

    #Angles: delete lines describing interactions with VS
    modified_lines_angles = []
    for line in list(angles_write.split("\n")):
        if line !="":
            angle_line = line.split()
            if line not in modified_lines_angles:
                modified_lines_angles.append(line)
            if len(angle_line)>2 and not line.startswith(";"):
                for vs, cb in virtual_sites.items():
                    if str(vs+1) in angle_line[:3] :
                        if line in modified_lines_angles : modified_lines_angles.remove(line)

    #Clean angles already described by dihedrals
    for lineA in modified_lines_angles:
        for lineD in list(dihedrals_write.split("\n")):
            angle_line = lineA.split()
            dihed_line = lineD.split()
            if len(dihed_line)>2 and not lineD.startswith(";") and len(angle_line)>2 and not lineA.startswith(";"):
                if angle_line[0] in dihed_line[:4] and angle_line[1] in dihed_line[:4] and angle_line[2] in dihed_line[:4]:
                    if lineA in modified_lines_angles : modified_lines_angles.remove(lineA)
    modified_angles_write = "\n".join(modified_lines_angles)

    if not simple_model:
        #Dihedrals: delete lines describing interactions with VS
        modified_lines_dihedrals = []
        dih_list = []
        for line in list(dihedrals_write.split("\n")):
            if line !="":
                dihed_line = line.split()
                if line not in modified_lines_dihedrals: modified_lines_dihedrals.append(line)
                if len(dihed_line)>2 and not line.startswith(";"):
                    dih_list.append(dihed_line[:4])
                    for vs, cb in virtual_sites.items():
                        if str(vs+1) in dihed_line[:4] :
                            if line in modified_lines_dihedrals : modified_lines_dihedrals.remove(line)
        for i in rigid_dih:
            modified_lines_dihedrals.append(i)

        modified_dihedrals_write = "\n".join(modified_lines_dihedrals)
    else:
        modified_dihedrals_write = dihedrals_write

    exclusions_net=""
    exclusions_net = exclusions_net + "\n[exclusions]\n"
    for i in range(1,nb_beads):
        row = " ".join(map(str, range(i, nb_beads + 1)))
        exclusions_net="   "+exclusions_net+row+"\n"


    #bartender info search

    bartender_input_info["VSITES"]=[]
    for line in list(vs_write.split("\n")):
        if ";" not in line and len(line.split())>3:
            info_vs = f"{line.split()[0]} {','.join(line.split()[2:])} 1"
            bartender_input_info["VSITES"].append(info_vs)

    bartender_input_info["BONDS"]=[]
    for line in list(modified_bonds_write.split("\n")):
        if ";" not in line and len(line.split())>4:
            bartender_input_info["BONDS"].append(line.split()[:2])

    bartender_input_info["ANGLES"]=[]
    for line in list(modified_angles_write.split("\n")):
        if ";" not in line and len(line.split())>5:
            bartender_input_info["ANGLES"].append(line.split()[:3])

    bartender_input_info["IMPROPERS"]=[]
    for line in list(dihedrals_write.split("\n")): # not modified_dihedrals_write
        if ";" not in line and len(line.split())>6:
            bartender_input_info["IMPROPERS"].append(line.split()[:4])

    text = modified_header_write +"\n"+ modified_atoms_write+"\n"+ modified_bonds_write+"\n"+ modified_angles_write+ "\n"+ modified_dihedrals_write + "\n"+ vs_write + exclusions_net
    return text, vs_bead_names, bartender_input_info


def bartender_input(mol, molname, atoms_in_beads, bart_info_dict):
    """Generates ready-to-use input data for Bartender"""
    text=f"# INPUT data for bonded parameter definition by BARTENDER for molecule {molname}\n"

    text+="BEADS\n"
    heavy_at=[]
    hydr_at=[]
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol()!='H':
            heavy_at.append(i)
        else:
            hydr_at.append(i)
    heavy_hydro_pair =[]

    for ib in range(len(mol.GetBonds())):
        abond = mol.GetBondWithIdx(ib)
        if (abond.GetBeginAtomIdx() in heavy_at and abond.GetEndAtomIdx() in hydr_at) or (abond.GetBeginAtomIdx() in hydr_at and abond.GetEndAtomIdx() in heavy_at):
            heavy_hydro_pair.append([abond.GetBeginAtomIdx(),abond.GetEndAtomIdx()])

    for bead,atomlist in atoms_in_beads.items():
        atoms=re.findall(r'\d+',atomlist)
        for pair in heavy_hydro_pair:
            if str(pair[0]) in atoms:
                atoms.append(pair[1])
            if str(pair[1]) in atoms:
                atoms.append(pair[0])
        incr_at=[int(atom) + 1 for atom in atoms]
        at_str=",".join(map(str, incr_at))
        text+=str(bead)+" "+at_str+"\n"

    for tp, info in bart_info_dict.items():
        if info:
            text+=tp+'\n'
            if tp=='VSITES':
                for i in info:
                    text+=f"{i}\n"
            else:
                for i in info:
                    text+=f"{','.join(i)}\n"
    return text
