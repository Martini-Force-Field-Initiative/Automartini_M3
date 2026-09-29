"""Molecule construction and atom/feature extraction.

Turns a SMILES string or an SD file into a 3D-embedded RDKit molecule, and
pulls out the basic per-atom information (heavy-atom list, ring membership,
aromaticity, H-bond features, coordinates) that the rest of the package
builds on.
"""

from ..common import *

logger = logging.getLogger(__name__)

# For feature extraction
fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def gen_molecule_smi(smi):
    """Generate mol object from smiles string"""
    logger.debug("Entering gen_molecule_smi()")
    errval = 0
    if "." in smi:
        logger.warning("Error. Only one molecule may be provided.")
        logger.warning(smi)
        errval = 4
        exit(1)
    # If necessary, adjust smiles for Aromatic Ns
    # Redirect current stderr in log file
    stderr_fd = None
    stderr_save = None
    try:
        stderr_fileno = sys.stderr.fileno()
        stderr_save = os.dup(stderr_fileno)
        stderr_fd = open("sanitize.log", "w")
        os.dup2(stderr_fd.fileno(), stderr_fileno)
    except Exception:
        stderr_fileno = None
    # Get smiles without sanitization
    molecule = Chem.MolFromSmiles(smi, False)
    try:
        cp = Chem.Mol(molecule)
        Chem.SanitizeMol(cp)

        # Close log file and restore old sys err
        if stderr_fileno is not None:
            stderr_fd.close()
            os.dup2(stderr_save, stderr_fileno)
        molecule = cp
    except ValueError:
        logger.warning("Bad smiles format %s found" % smi)
        nm = AdjustAromaticNs(molecule)

        if nm is not None:
            Chem.SanitizeMol(nm)
            molecule = nm
            smi = Chem.MolToSmiles(nm)
            logger.warning("Fixed smiles format to %s" % smi)
        else:
            logger.warning("Smiles cannot be adjusted %s" % smi)
            errval = 1
    # Continue
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=1, useRandomCoords=True)  # Set Seed for random coordinate generation = 1.
    try:
        AllChem.UFFOptimizeMolecule(molecule)
    except ValueError as e:
        logger.warning("%s" % e)
        exit(1)
    return molecule, errval


def gen_molecule_sdf(sdf):
    """Generate mol object from SD file"""
    logger.debug("Entering gen_molecule_sdf()")
    suppl = Chem.SDMolSupplier(sdf)
    if len(suppl) > 1:
        print("Error. Only one molecule may be provided.")
        exit(1)
    molecule = ""
    for molecule in suppl:
        if molecule is None:
            print("Error. Can't read molecule.")
            exit(1)
    Chem.SanitizeMol(molecule)
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=1, useRandomCoords=True)  # Set Seed for random coordinate generation = 1.
    try:
        AllChem.UFFOptimizeMolecule(molecule)
    except ValueError as e:
        exit(1)
    return molecule


def heavy_atoms_first(molecule):
    """Return the molecule with its atoms renumbered so the heavy atoms come
    first, in their original relative order, followed by the hydrogens.

    The mapping code uses a heavy atom's position in the heavy-atom list
    (0..n-1) as its atom index, which only holds when no hydrogen sits between
    heavy atoms. Hydrogens added by Chem.AddHs are appended at the end, but
    hydrogens written explicitly in the input (e.g. SMILES "[H]OCC") keep their
    place. A molecule already in that order is returned unchanged.
    """
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() != "H"]
    if heavy == list(range(len(heavy))):
        return molecule
    hydrogens = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "H"]
    return Chem.RenumberAtoms(molecule, heavy + hydrogens)


def get_charge(molecule):
    """Get net charge of molecule"""
    logger.debug("Entering get_charge()")
    return Chem.rdmolops.GetFormalCharge(molecule)


def get_hbond_a(features):
    """Get Hbond acceptor information"""
    logger.debug("Entering get_hbond_a()")
    hbond = []
    for feat in features:
        if feat.GetFamily() == "Acceptor":
            for i in feat.GetAtomIds():
                if i not in hbond:
                    hbond.append(i)
    return hbond


def get_hbond_d(features):
    """Get Hbond donor information"""
    logger.debug("Entering get_hbond_d()")
    hbond = []
    for feat in features:
        if feat.GetFamily() == "Donor":
            for i in feat.GetAtomIds():
                if i not in hbond:
                    hbond.append(i)
    return hbond


def get_atoms(molecule):
    """List all heavy atoms"""
    logger.debug("Entering get_atoms()")
    conformer = molecule.GetConformer()
    num_atoms = conformer.GetNumAtoms()
    list_heavyatoms = []
    list_heavyatomnames = []

    atoms = np.arange(num_atoms)
    for i in np.nditer(atoms):
        atom_name = molecule.GetAtomWithIdx(int(atoms[i])).GetSymbol()
        if atom_name != "H":
            list_heavyatoms.append(atoms[i])
            list_heavyatomnames.append(atom_name)

    if len(list_heavyatoms) == 0:
        print("Error. No heavy atom found.")
        exit(1)
    return list_heavyatoms, list_heavyatomnames


def get_ring_atoms(mol):
    """Get ring atoms and systems of joined (fused/bridged) rings"""
    logger.debug("Entering get_ring_atoms()")

    rings = mol.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in rings:
        ring_atoms = set(ring)
        new_systems = []
        for system in ring_systems:
            shared = len(ring_atoms.intersection(system))
            if shared:
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems

    return [list(ring) for ring in ring_systems]


def is_aromatic(mol):
    """Returns (has_aromatic_atoms, num_aromatic_atoms) for the molecule"""
    aromatic_atoms = [atom.GetIsAromatic() for atom in mol.GetAtoms()]
    num_aromatic_atoms = sum(aromatic_atoms)
    return (num_aromatic_atoms > 0, num_aromatic_atoms)


def get_heavy_atom_coords(molecule):
    """Extract atomic coordinates of heavy atoms in molecule mol"""
    logger.debug("Entering get_heavy_atom_coords()")
    heavyatom_coords = []
    allatom_coords = []
    conformer = molecule.GetConformer()
    # number of atoms in mol
    num_atoms = molecule.GetConformer().GetNumAtoms()
    for i in range(num_atoms):
        if molecule.GetAtomWithIdx(i).GetSymbol() != "H":
            heavyatom_coords.append(np.array([conformer.GetAtomPosition(i)[j] for j in range(3)]))
            allatom_coords.append(np.array([conformer.GetAtomPosition(i)[j] for j in range(3)]))
        else:
            allatom_coords.append(np.array([conformer.GetAtomPosition(i)[j] for j in range(3)]))
    return conformer, heavyatom_coords, allatom_coords


def extract_features(molecule):
    """Extract features of molecule"""
    logger.debug("Entering extract_features()")
    features = factory.GetFeaturesForMol(molecule)
    return features
