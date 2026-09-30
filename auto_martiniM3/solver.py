"""
Created on March 13, 2019 by Andrew Abi-Mansour
Updated to Martini 3 force field on January 31, 2025 by Magdalena Szczuka

This is the::
    _   _   _ _____ ___     __  __    _    ____ _____ ___ _   _ ___   __  __ _____
   / \ | | | |_   _/ _ \   |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|  |  \/  |___ /  
  / _ \| | | | | || | | |  | |\/| | / _ \ | |_) || |  | ||  \| || |   | |\/| | |_ \  
 / ___ \ |_| | | || |_| |  | |  | |/ ___ \|  _ < | |  | || |\  || |   | |  | |___) | 
/_/  _\_\___/  |_| \___/   |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|  |_|  |_|____/    
                                                

A tool for automatic MARTINI 3 force field mapping and parametrization of small organic molecules

Developers::
        Magdalena Szczuka (magdalena.szczuka at univ-tlse3.fr)
        Tristan BEREAU (bereau at mpip-mainz.mpg.de)
        Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
        Andrew Abi-Mansour (andrew.gaam at gmail.com)

AUTO_MARTINI M3 is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
""" 

from dataclasses import dataclass

from . import optimization, output, topology
from .common import *

logger = logging.getLogger(__name__)

def get_coords(conformer, sites, avg_pos, ringatoms_flat):
    """Extract coordinates of CG beads"""
    logger.debug("Entering get_coords()")
    # CG beads are averaged over best trial combinations for all
    # non-aromatic atoms.
    logger.debug("Entering get_coords()")
    site_coords = []
    for i in range(len(sites)):
        if sites[i] in ringatoms_flat:
            site_coords.append(
                np.array([conformer.GetAtomPosition(int(sites[i]))[j] for j in range(3)])
            )
        else:
            # Use average
            site_coords.append(np.array(avg_pos[i]))
    return site_coords


def check_additivity(forcepred, beadtypes, molecule, mol_smi): #AutoM3 change : added mol_smi argument
    """Check additivity assumption between sum of free energies of CG beads
    and free energy of whole molecule"""
    logger.debug("Entering check_additivity()")
    # If there's only one bead, don't check.
    sum_frag = 0.0
    rings = False
    logger.info("; Bead types: %s" % beadtypes)
    for bead in beadtypes:
        if bead[0] == "S" or bead[0] == "T": # AutoM3 change : added bead "T"
            rings = True
        delta_f_types = topology.read_delta_f_types()
        sum_frag += delta_f_types[bead] #sum of free energies of beads in ring(s)
    # Wildman-Crippen log_p
    wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(molecule)[0]
    # Get SMILES string of entire molecule

    whole_mol_dg,_ = topology.smi2alogps(forcepred, mol_smi, wc_log_p, "MOL",None,None,True) # AutoM3 change : None,None=converted_smi, real_smi not needed here
    if whole_mol_dg != 0:
        m_ad = math.fabs((whole_mol_dg - sum_frag) / whole_mol_dg)
        logger.info(
            "; Mapping additivity assumption ratio: %7.4f (whole vs sum: %7.4f vs. %7.4f)"
            % (m_ad, whole_mol_dg / (-4.184), sum_frag / (-4.184))
        )
        if len(beadtypes) == 1:
            return True
        if (not rings and m_ad < 0.5) or rings:
            return True
        else:
            return False
    else:
        return False


@dataclass
class _RunOptions:
    """What the caller asked Cg_molecule for (names, prediction options, output files)."""
    molname: str
    mol_smi: str
    simple_model: bool
    topfname: str
    bartenderfname: str
    bartender: bool
    logp_file: str
    forcepred: bool


class _MoleculeData:
    """Everything the mapping needs to know about the embedded molecule, computed once."""

    def __init__(self, molecule):
        self.molecule = molecule
        features = topology.extract_features(molecule)
        self.heavy_atoms, self.heavy_atom_names = topology.get_atoms(molecule)
        self.conformer, self.heavy_atom_coords, self.atom_coords = topology.get_heavy_atom_coords(molecule)
        # Ring systems (fused rings merged), aromaticity and H-bond donors/acceptors
        self.ring_atoms = topology.get_ring_atoms(molecule)
        self.is_arom, self.num_arom = topology.is_aromatic(molecule)
        self.hbond_a = topology.get_hbond_a(features)
        self.hbond_d = topology.get_hbond_d(features)
        self.ring_atoms_flat = list(chain.from_iterable(self.ring_atoms))
        # Pairing template of the fused ring systems (see optimization.fused_ring_groups)
        self.ring_groups = optimization.fused_ring_groups(molecule)


class Cg_molecule:
    """Main class to coarse-grain molecule.

    Building it runs the whole mapping: embed the molecule in 3D, score every
    candidate bead placement (optimization.find_bead_pos), then try the
    candidates from best to worst score until one passes all the checks, and
    write its topology.
    """

    def __init__(self, molecule, mol_smi, molname, simple_model, topfname, bartenderfname, bartender, logp_file, forcepred=True):
        # AutoM3 new arguments : mol_smi, simple_model, bartenderfname, bartender, logp_file

        self.heavy_atom_coords = None
        self.atom_coords = None # AutoM3 new variable
        self.list_heavyatom_names = None
        self.atom_partitioning = None
        self.cg_bead_names = []
        self.cg_bead_coords = []
        self.topout = None
        self.bartender_out = None # AutoM3 new variable
        self.molname=molname # AutoM3 change : for pretty GRO file (will be easier to look on a molecule in VMD with its proper name)

        logger.info("Entering cg_molecule()")

        options = _RunOptions(molname, mol_smi, simple_model, topfname, bartenderfname, bartender, logp_file,
                              forcepred)
        mol = _MoleculeData(self._embed(molecule))
        self.list_heavyatom_names = mol.heavy_atom_names
        self.heavy_atom_coords, self.atom_coords = mol.heavy_atom_coords, mol.atom_coords
        self._find_mapping(mol, options)

    @staticmethod
    def _embed(molecule):
        """3D structure the mapping works on: heavy atoms first, embedded, MMFF94s-minimized."""
        ### AutoM3 : MINIMIZATION with RDkit ###
        molecule = topology.heavy_atoms_first(Chem.Mol(molecule))
        # Unseeded, RDKit draws from a process-wide generator, so the conformer (and
        # thus the mapping) depended on what had been embedded earlier in the same
        # process. 42 is that generator's initial seed: a molecule run on its own
        # (e.g. one CLI call) gets exactly the conformer it got before.
        AllChem.EmbedMolecule(molecule, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000,mmffVariant='MMFF94s')
        return molecule

    def _find_mapping(self, mol, options):
        """Try the candidate bead placements from best to worst score until one
        passes every check, then write its topology.

        The first pass tolerates no failed check. If no candidate among the best
        half passes, the same candidates are tried again in force_map mode: older
        partitioning, and one failed check tolerated.
        """
        # Optimize coarse-grained bead positions -- keep all possibilities in case something goes
        # wrong later in the code.
        list_cg_beads, list_bead_pos = optimization.find_bead_pos(
            mol.molecule, mol.conformer, mol.heavy_atoms, self.heavy_atom_coords, self.atom_coords,
            mol.ring_atoms, mol.ring_atoms_flat, False,
        )

        # A bead holds at most 2 ring atoms, so that limit can only be met by a candidate with
        # at least ceil(ring atoms / 2) beads. When no candidate is that large (e.g. anthracene,
        # bithiophene) every candidate would fail it until the force_map fallback, so skip it.
        ring_limit_reachable = (max((len(c) for c in list_cg_beads), default=0)
                                >= math.ceil(len(set(mol.ring_atoms_flat)) / 2))

        # Loop through best 1% cg_beads and avg_pos
        max_attempts = int(math.ceil(0.5 * len(list_cg_beads)))
        logger.info(f"Max. number of attempts: {max_attempts}")
        force_map = False
        attempt = 0
        while attempt < max_attempts:
            cg_beads = list_cg_beads[attempt]
            passed, cg_beads_rings = self._check_candidate(
                mol, options, cg_beads, list_bead_pos[attempt], list_cg_beads[0], force_map, ring_limit_reachable
            )
            if passed:
                self._write_topology(mol, options, cg_beads, cg_beads_rings)
                print("Converged to solution in {} iteration(s)".format(
                    attempt + 1 + (max_attempts if force_map else 0)))
                return
            attempt += 1
            # AutoM3 change : force mapping by old code if new code doesn't give result
            if attempt == max_attempts and not force_map:
                force_map = True
                attempt = 0

        if attempt == max_attempts and force_map:
            raise RuntimeError(
                "ERROR: no successful mapping found.\nTry running with the --fpred and/or --verbose options."
            )

    def _check_candidate(self, mol, options, cg_beads, bead_pos, best_cg_beads, force_map, ring_limit_reachable):
        """Partition the atoms between the beads of one candidate and run every check on it.

        Sets atom_partitioning, cg_bead_coords and cg_bead_names for this candidate.
        Returns (passed, cg_beads_rings).
        """
        success = True
        # Remove mappings with bead numbers less than most optimal mapping.
        if len(cg_beads) < len(best_cg_beads) and (len(mol.heavy_atoms) - (5 * len(cg_beads))) > 3:
            success = False

        # AutoM3 : newer partitioning for molecules with few aromatic atoms, the older one otherwise
        cg_bead_coords = get_coords(mol.conformer, cg_beads, bead_pos, mol.ring_atoms_flat)
        voronoi = (optimization.voronoi_atoms_new if not force_map and mol.num_arom < 7
                   else optimization.voronoi_atoms_old)
        self.atom_partitioning, self.cg_bead_coords = voronoi(
            cg_bead_coords, self.heavy_atom_coords, self.atom_coords, mol.molecule, ring_groups=mol.ring_groups
        )

        # AutoM3 checks: at most 2 ring atoms per bead and the fused-ring template, and
        # functional groups kept whole. force_map tolerates one failed check (and then
        # overrides the bead-number rule above); otherwise none may fail.
        # The 2-ring-atoms-per-bead limit only applies with an even number of aromatic
        # atoms (an odd ring can't be split that way) and when some candidate has enough
        # beads for it; the fused-ring template applies to any fused system, aromatic or not.
        fails = 0
        if mol.is_arom or mol.ring_groups:
            if not optimization.max2arperbead(self.atom_partitioning, mol.ring_atoms, ring_groups=mol.ring_groups,
                                              ring_atom_limit=mol.is_arom and (mol.num_arom % 2) == 0
                                              and ring_limit_reachable):
                fails += 1
        if not optimization.functional_groups_ok(self.atom_partitioning, mol.molecule, mol.ring_atoms):
            fails += 1
        if force_map:
            success = fails <= 1
        elif fails > 0:
            success = False

        logger.info("; Atom partitioning: {atom_partitioning}")

        # cgbeads should take atom rings number if ring atom in bead
        cg_beads_rings = cg_beads.copy()
        for i, b in enumerate(cg_beads):
            if b not in mol.ring_atoms_flat:
                atoms_in_b = [at for at, bd in self.atom_partitioning.items() if bd == i]
                for a in atoms_in_b:
                    if a in mol.ring_atoms_flat:
                        cg_beads_rings[i] = a

        # Bead types, and the additivity check between fragments and entire molecule
        self.cg_bead_names, bead_types, _, _ = topology.print_atoms(
            options.molname, options.forcepred, cg_beads, mol.molecule, mol.hbond_a, mol.hbond_d,
            self.atom_partitioning, mol.ring_atoms, mol.ring_atoms_flat, options.logp_file, True,
        )
        if not self.cg_bead_names:
            success = False
        if not check_additivity(options.forcepred, bead_types, mol.molecule, options.mol_smi):
            success = False

        # Bond count: a tree of beads (or more links with rings), and one name per bead
        bond_list, const_list, _ = topology.print_bonds(
            cg_beads, cg_beads_rings, mol.molecule, self.atom_partitioning, self.cg_bead_coords, bead_types,
            mol.ring_atoms, trial=True,
        )
        num_links = len(bond_list) + len(const_list)
        if not mol.ring_atoms and num_links >= len(self.cg_bead_names):
            success = False
        if num_links < len(self.cg_bead_names) - 1:
            success = False
        if len(cg_beads) != len(self.cg_bead_names):
            success = False
        return success, cg_beads_rings

    def _write_topology(self, mol, options, cg_beads, cg_beads_rings):
        """Build the topology of the accepted candidate (self.topout) and write the requested files."""
        header_write = topology.print_header(options.molname, options.mol_smi)
        self.cg_bead_names, bead_types, atoms_write, atoms_in_smi = topology.print_atoms( # AutoM3 new variable : atoms_in_smi
            options.molname, options.forcepred, cg_beads, mol.molecule, mol.hbond_a, mol.hbond_d,
            self.atom_partitioning, mol.ring_atoms, mol.ring_atoms_flat, options.logp_file, trial=False,
        )
        bond_list, const_list, bonds_write = topology.print_bonds(
            cg_beads, cg_beads_rings, mol.molecule, self.atom_partitioning, self.cg_bead_coords, bead_types,
            mol.ring_atoms, False,
        )
        if not options.simple_model: # AutoM3
            dihedrals_write = topology.print_dihedrals(
                cg_beads, const_list, mol.ring_atoms, self.cg_bead_coords, bead_types
            )
        angles_write, angle_list = topology.print_angles(
            cg_beads, mol.molecule, self.atom_partitioning, self.cg_bead_coords, bead_types, bond_list, const_list,
            mol.ring_atoms,
        )
        # AutoM3 change : possible simple output w/o dihedrals, virtual sites
        self.topout, bartender_input_info = topology.topout(header_write, atoms_write, bonds_write, angles_write)

        # Fused ring systems (or rings larger than 6) get virtual sites
        if len(mol.ring_atoms) > 1:
            common = (len(set.intersection(*map(set, mol.ring_atoms))) > 1
                      or any(len(system) > 6 for system in mol.ring_atoms))
        else:
            common = len(mol.ring_atoms_flat) > 6

        ### AutoM3 outputs ###
        if len(mol.ring_atoms_flat) > 0 and not options.simple_model:
            if len(mol.ring_atoms_flat) > 7 and common:
                vs_write, virtual_sites, rigid_dih = topology.print_virtualsites(
                    mol.ring_atoms, self.cg_bead_coords, self.atom_partitioning, mol.molecule
                )
                self.topout, vs_bead_names, bartender_input_info = topology.topout_vs(
                    header_write, atoms_write, bonds_write, angles_write, dihedrals_write, virtual_sites, vs_write,
                    rigid_dih, options.simple_model,
                )
            else:
                self.topout, bartender_input_info = topology.topout_noVS(
                    header_write, atoms_write, bonds_write, angles_write, dihedrals_write, self.cg_bead_coords,
                    mol.ring_atoms, cg_beads,
                )

        if options.bartender:
            bartender_out = topology.bartender_input(mol.molecule, options.molname, atoms_in_smi, bartender_input_info)
            with open(options.bartenderfname, "w") as btf:
                btf.write(bartender_out)
        if options.topfname:
            with open(options.topfname, "w") as fp:
                fp.write(self.topout)

    def output_aa(self, aa_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional all-atom output to GRO file
        aa_out = output.output_gro(self.heavy_atom_coords, self.list_heavyatom_names, self.molname)
        if aa_output:
            with open(aa_output, "w") as fp:
                fp.write(aa_out)
        else:
            return aa_out

    def output_cg(self, cg_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional coarse-grained output to GRO file
        cg_out = output.output_gro(self.cg_bead_coords, self.cg_bead_names, self.molname)
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out
