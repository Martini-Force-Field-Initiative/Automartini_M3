"""Martini 3 bead-type assignment.

Given a CG-bead fragment (its SMILES, charge, H-bond character, ring
membership), determines the closest Martini 3 bead type by comparing
free-energy-of-transfer (delta_f) values, and holds the reference bonded
force-field parameters (bond/angle/dihedral force constants) used to write
the topology out.
"""

from ..common import *

logger = logging.getLogger(__name__)


def read_delta_f_types():
    """
    Delta_f types for Martini 3, from SI https://doi.org/10.1038/s41592-021-01098-3
    Returns delta_f types dictionary
    """
    delta_f_types = dict()
    delta_f_types = {"C1":18.9,"C2":14.8,"C3":13.8,"C4":13.4,"C5":11.2,"C6":10.1,"N1":8.1,"N2":5.6,"N3":1.8,"N4":2.2,"N5":0.0,"N6":-1.1,"P1":-2.0,"P2":-3.8,"P3":-5.1,"P4":-7.4,"P5":-9.1,
                     "P6":-9.2,"X1":14.3,"X2":12.7,"X3":13.9,"X4":8.7,"N1d":10.7,"N1a":10.7,"N2d":7.8,"N2a":7.8,"N3d":3.8,"N3a":3.8,"N4d":4.3,"N4a":4.3,"N5d":2.2,"N5a":2.2,"N6d":1.0,
                     "N6a":1.0,"P1d":0.2,"P1a":0.2,"P2a":-1.9,"P2d":-1.9,"P3d":-3.5,"P3a":-3.5,"P4d":-5.1,"P4a":-5.1,"P5d":-7.0,"P5a":-7.0,"P6d":-7.4,"P6a":-7.4,"Q1":-10.9,"Q2":-15.1,
                     "Q3":-17.4,"Q4":-18.8,"Q5":-23.0,"D":-26.8,
                     "SC1":14.2,"SC2":9.9,"SC3":9.2,"SC4":8.4,"SC5":6.3,"SC6":5.3,"SN1":3.6,"SN2":2.1,"SN3":-1.8,"SN4":-0.9,"SN5":-3.6,"SN6":-4.2,"SP1":-5.2,"SP2":-6.9,"SP3":-7.7,"SP4":-9.8,"SP5":-11.8,
                     "SP6":-12.0,"SX1":9.4,"SX2":7.2,"SX3":8.0,"SX4":4.3,"SN1d":6.0,"SN1a":6.0,"SN2d":3.8,"SN2a":3.8,"SN3d":0.2,"SN3a":0.2,"SN4d":1.1,"SN4a":1.1,"SN5d":-1.0,"SN5a":-1.0,"SN6d":-2.5,
                     "SN6a":-2.5,"SP1d":-3.7,"SP1a":-3.7,"SP2d":-5.4,"SP2a":-5.4,"SP3d":-6.1,"SP3a":-6.1,"SP4d":-7.8,"SP4a":-7.8,"SP5d":-9.5,"SP5a":-9.5,"SP6d":-9.6,"SP6a":-9.6,"SQ1":-10.6,"SQ2":-14.3,
                     "SQ3":-18.0,"SQ4":-18.2,"SQ5":-18.2,"SD":-36.4,
                     "TC1":12.0,"TC2":7.8,"TC3":6.7,"TC4":6.4,"TC5":4.5,"TC6":3.6,"TN1":2.3,"TN2":0.3,"TN3":-3.1,"TN4":-2.9,"TN5":-4.9,"TN6":-6.1,"TP1":-7.2,"TP2":-8.8,"TP3":-9.8,"TP4":-12.1,"TP5":-15.2,
                     "TP6":-14.8,"TX1":7.6,"TX2":5.2,"TX3":5.4,"TX4":2.7,"TN1d":3.9,"TN1a":3.9,"TN2d":2.3,"TN2a":2.3,"TN3d":-1.4,"TN3a":-1.4,"TN4d":-1.2,"TN4a":-1.2,"TN5d":-2.8,"TN5a":-2.8,"TN6d":-4.1,
                     "TN6a":-4.1,"TP1d":-5.0,"TP1a":-5.0,"TP2d":-6.8,"TP2a":-6.8,"TP3d":-7.8,"TP3a":-7.8,"TP4d":-9.5,"TP4a":-9.5,"TP5d":-13.2,"TP5a":-13.2,"TP6d":-12.7,"TP6a":-12.7,"TQ1":-14.2,"TQ2":-14.5,
                     "TQ3":-18.7,"TQ4":-16.3,"TQ5":-17.0,"TD":-36.8
                     }
    return delta_f_types


def letter_occurrences(string):
    """Count letter occurences"""
    logger.debug("Entering letter_occurrences()")
    frequencies = defaultdict(lambda: 0)
    for character in string:
        if character.isalnum():
            frequencies[character.upper()] += 1
    return frequencies


def cyclic_smi_conversion(smi):
    """Convert cyclic atoms in a SMILES string from lower case to upper case
    so rdkit accepts it without raising:
    rdkit.Chem.rdchem.AtomKekulizeException: non-ring atom 0 marked aromatic
    """
    smi = smi.replace("ccc","CC=C")
    smi = smi.replace("cc","C=C")
    smi = smi.replace("c","C")
    smi = smi.replace("n","N")
    smi = smi.replace("s","S")
    smi = smi.replace("o","O")
    return (smi)


def find_closest_key(dictionary, target_value):
    """Return the dict key numerically closest to target_value"""
    lst=list(dictionary.keys())
    closest_key = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-target_value))]
    return closest_key


def rearrange_until_match(input_string):
    letters = [char for char in input_string if char.isalpha()]
    random.shuffle(letters)
    result_string = '-'.join(letters)
    return result_string


def read_params(val, size):
    """Returns the closest force value to given parameter, based on
    state-of-the-art parametrizations available in MAD (https://mad.ibcp.fr)"""
    bonds = {'S-S': {0.36: 5000.0, 0.378: 5000.0, 0.321: 25000.0, 0.331: 5000.0, 0.3: 5000.0, 0.37: 5000.0, 0.281: 25000.0,
                     0.314: 25000.0, 0.32: 7500.0, 0.38: 5000.0, 0.33: 17000.0, 0.405: 5000.0, 0.395: 5000.0, 0.39: 5000.0,
                     0.385: 5000.0, 0.35: 5000.0, 0.375: 3500.0, 0.376: 7000.0, 0.34: 7000.0},
             'T-T': {0.32: 25000.0, 0.261: 25000.0, 0.376: 25000.0, 0.25: 25000.0, 0.401: 25000.0, 0.449: 100000.0,
                     0.251: 100000.0},
             'T-S': {0.364: 25000.0, 0.408: 25000.0, 0.272: 25000.0, 0.31: 7000.0, 0.3: 5000.0, 0.253: 5000.0, 0.387: 25000.0,
                     0.34: 5000.0, 0.29: 5000.0, 0.32: 5000.0, 0.33: 10000.0, 0.286: 100000.0, 0.371: 100000.0, 0.244: 100000.0,
                     0.355: 5000.0, 0.36: 5000.0},
             'R-T': {0.389: 5000.0},
             'R-R': {0.38: 50000.0, 0.475: 3800.0, 0.47: 3800.0, 0.468: 3800.0, 0.49: 3800.0, 0.46: 7000.0, 0.45: 7000.0,
                     0.455: 7000.0},
             'R-S': {0.385: 7000.0, 0.38: 7000.0, 0.405: 7000.0}
             }

    angles = {'T-T-S': {180.0: 250.0, 138.0: 250.0, 71.0: 250.0, 122.0: 50.0},
              'T-S-S': {155.0: 100.0, 148.0: 100.0},
              'T-S-T': {135.0: 30.0},
              'S-S-S': {150.6: 100.0, 130.0: 25.0, 150.0: 100.0, 135.0: 15.0},
              'T-T-R': {160.0: 180.0},
              'R-R-R': {180.0: 35.0, 100.0: 10.0},
              }

    dihedrals = {'S-S-S-T': {180.0: 100.0},
                 'S-S-T-T': {180.0: 100.0},
                 'S-T-T-S': {180.0: 75.0, 0.0: 50.0},
                 'S-T-T-T': {180.0: 200.0, 0.0: 100.0},
                 'T-T-T-T': {180.0: 200.0, 0.0: 100},
                 'T-S-S-T': {180.0: 100.0},
                 'R-T-T-T': {180.0: 50.0},
                 'T-T-S-S': {180.0: 50.0},
                 'S-T-S-S': {0.0: 50.0},
                 'T-T-T-S': {180.0: 20.0},
                 'T-R-R-T': {0.0: 1.8},
                 'T-T-S-T': {-45.0: 200.0},
                 'S-S-S-S': {180.0: 1.96, 0.0: 0.18}}


    if len(size)==3: #bonds
        if size not in bonds.keys(): size = size[2]+'-'+size[0]
        for k,v in bonds.items():
            if k == size:
                closest_length = find_closest_key(v, val)
                force = v[closest_length]
                return force

    if len(size)==5: #angles
        key_exists=False
        if size in angles.keys():
            for k,v in angles.items():
                if k == size:
                    closest_length = find_closest_key(v, val)
                    force = v[closest_length]
                    return force

    if len(size)==7: #dihedrals
        key_exists=False
        if size in dihedrals.keys():
            for k,v in dihedrals.items():
                if k == size:
                    closest_length = find_closest_key(v, val)
                    force = v[closest_length]
                    return force


def substruct2smi(molecule, partitioning, cg_bead):
    """Substructure to smiles conversion; also output Wildman-Crippen log_p;
    and charge of group."""
    frag = rdchem.EditableMol(molecule)
    # fragment smi: [H]N([H])c1nc(N([H])[H])n([H])n1
    num_atoms = molecule.GetConformer().GetNumAtoms()

    # First delete all hydrogens
    for i in range(num_atoms):
        if molecule.GetAtomWithIdx(i).GetSymbol() == "H":
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetConformer().GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(i)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    # Then heavy atoms that aren't part of the CG bead (except those
    # involved in the same ring).
    for i in partitioning.keys():
        if partitioning[i] != cg_bead:
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetConformer().GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(i)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    # Wildman-Crippen log_p
    wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(frag.GetMol())[0]

    # Charge -- look at atoms that are only part of the bead (no ring rule)
    chg = 0
    for i in partitioning.keys():
        if partitioning[i] == cg_bead:
            chg += molecule.GetAtomWithIdx(i).GetFormalCharge()

    smi = Chem.MolToSmiles(Chem.rdmolops.AddHs(frag.GetMol(), addCoords=True))

    atoms_in_smi=" ; atoms: "
    converted_smi=False
    real_smi=None

    for at, bd in partitioning.items():
        if bd == cg_bead:
            at_symbol = molecule.GetAtomWithIdx(at).GetSymbol()
            atoms_in_smi += at_symbol + str(at) + ", "

    if "c" in smi or "n" in smi or "s" in smi:
        converted_smi = True
        real_smi=smi
        smi = cyclic_smi_conversion(smi)

    # fragment smi: Nc1ncnn1 ---------> FAILURE! Need to fix this Andrew! For now, just a hackish soln:
    # smi = smi.lower() if smi.islower() else smi.upper()
    return smi, wc_log_p, chg, atoms_in_smi,converted_smi,real_smi


def get_mass(smi):
    """Gets real mass of atoms in smile code"""
    smi_mass=0
    atom_mass={"C":12,"O":16,"N":14,"S":32,"Cl":35,"I":127,"F":19,"Br":80,"P":31,"Si":28,"B":11,"Be":9,"Li":1,"Mg":24,"Ca":40,"K":39}
    i = 0
    while i < len(smi):
        if i < len(smi)-1 and smi[i:i+2] in atom_mass:  # Check if the current two characters form a known atom
            smi_mass += atom_mass[smi[i:i+2]]
            i += 2
        elif smi[i] in atom_mass:  # Check if the current character forms a known atom
            smi_mass += atom_mass[smi[i]]
            i += 1
        else:  # Skip unknown characters
            i += 1
    return smi_mass


def get_standard_mass(bead_type):
    """Gets standard mass of atoms in smile code"""
    if bead_type.startswith('T'): return 36
    else:
        if bead_type.startswith('S'): return 54
        else: return 72


def mad(bead_type, delta_f, in_ring=False):
    """Mean absolute difference between bead type and delta_f"""
    delta_f_types = read_delta_f_types()
    return math.fabs(delta_f_types[bead_type] - delta_f)


def count_letters(s):
    """Counting atoms in SMILES code"""
    count = 0
    i = 0
    while i < len(s):
        if s[i:i+2] in ["Cl", "Br"]:
            count += 1
            i += 2
        elif s[i].isalpha():
            count += 1
            i += 1
        else:
            i += 1
    return count


def find_closest_logPvalue(value, keyslist,in_ring):
    closest_key = None
    closest_diff = float('inf')
    dict=read_delta_f_types()
    for key in keyslist:
        if key in dict:
            diff = mad(key,value,in_ring)
            if diff < closest_diff:
                closest_key = key
                closest_diff = diff
    return closest_key


def determine_bead_type(delta_f, charge, hbonda, hbondd, in_ring, smi_frag):
    """Determine CG bead type from delta_f value, charge,
    and hbond acceptor, and donor"""
    if charge < -1 or charge > +1:
        print("Charge is too large: %s" % charge)
        exit(1)
    bead_type = []
    if charge != 0:
        # The compound has a +/- charge -> Q type

        if count_letters(str(smi_frag)) == 2:
            othertypes_Q=["TQ1","TQ2","TQ3","TQ4","TQ5","TD"]
        if count_letters(str(smi_frag)) == 3:
            othertypes_Q=["SQ1","SQ2","SQ3","SQ4","SQ5","SD"]
        if count_letters(str(smi_frag)) > 3:
            othertypes_Q=["Q1","Q2","Q3","Q4","Q5","D"]
        bead_type=find_closest_logPvalue(delta_f, othertypes_Q,in_ring)

    else:
        # Neutral group
        if hbonda > 0 or hbondd > 0:
            if count_letters(str(smi_frag)) == 2:
                othertypes_NPa=["TN1a","TN2a","TN3a","TN4a","TN5a","TN6a","TP1a","TP2a","TP3a","TP4a","TP5a","TP6a"]
                othertypes_NPd=["TN1d","TN2d","TN3d","TN4d","TN5d","TN6d","TP1d","TP2d","TP3d","TP4d","TP5d","TP6d"]
            if count_letters(str(smi_frag)) == 3:
                othertypes_NPa=["SN1a","SN2a","SN3a","SN4a","SN5a","SN6a","SP1a","SP2a","SP3a","SP4a","SP5a","SP6a"]
                othertypes_NPd=["SN1d","SN2d","SN3d","SN4d","SN5d","SN6d","SP1d","SP2d","SP3d","SP4d","SP5d","SP6d"]
            if count_letters(str(smi_frag)) > 3:
                othertypes_NPa=["N1a","N2a","N3a","N4a","N5a","N6a","P1a","P2a","P3a","P4a","P5a","P6a"]
                othertypes_NPd=["N1d","N2d","N3d","N4d","N5d","N6d","P1d","P2d","P3d","P4d","P5d","P6d"]

            if hbonda > 0 and hbondd == 0:
                bead_type=find_closest_logPvalue(delta_f, othertypes_NPa,in_ring)
            if hbonda  >= 0 and hbondd > 0:
                bead_type=find_closest_logPvalue(delta_f, othertypes_NPd,in_ring)

        else:
            # all other cases. Simply find the atom type that's closest in
            # free energy.

            if count_letters(str(smi_frag)) == 2:
                othertypes = ["TP6","TP5","TP4","TP3","TP2","TP1","TC6","TC5","TC4","TC3","TC2","TC1","TN6","TN5","TN4","TN3","TN2","TN1"]
                if not in_ring: othertypes.remove("TC5")

            if count_letters(str(smi_frag)) == 3:
                othertypes = ["SP6","SP5","SP4","SP3","SP2","SP1","SC6","SC5","SC4","SC3","SC2","SC1","SN6","SN5","SN4","SN3","SN2","SN1"]

            if count_letters(str(smi_frag)) > 3:
                othertypes = ["P6","P5","P4","P3","P2","P1","C6","C5","C4","C3","C2","C1","N6","N5","N4","N3","N2","N1"]

            bead_type=find_closest_logPvalue(delta_f, othertypes,in_ring)

    for hal in ["Cl","Br","F","I"]:
        if hal in str(smi_frag):
            if count_letters(str(smi_frag)) == 2: othertypes = ["TX4","TX3","TX2","TX1"]
            if count_letters(str(smi_frag)) == 3: othertypes = ["SX4","SX3","SX2","SX1"]
            if count_letters(str(smi_frag)) > 3: othertypes = ["X4","X3","X2","X1"]
            bead_type=find_closest_logPvalue(delta_f, othertypes,in_ring)

    return bead_type
