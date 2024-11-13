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


class Cg_molecule:
    """Main class to coarse-grain molecule"""

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
        force_map = False # AutoM3 new variable

        logger.info("Entering cg_molecule()")

        ### AutoM3 : MINIMIZATION with RDkit ###
        molecule = Chem.Mol(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000,mmffVariant='MMFF94s')
        AllChem.NormalizeDepiction(molecule, scaleFactor=1.12) #was 1.12

        feats = topology.extract_features(molecule)

        # Get list of heavy atoms and their coordinates
        list_heavy_atoms, self.list_heavyatom_names = topology.get_atoms(molecule)

        conf, self.heavy_atom_coords = topology.get_heavy_atom_coords(molecule)

        _, self.atom_coords = topology.get_atom_coords(molecule) # AutoM3 : for new voronoi partitioning function

        # Identify ring-type atoms
        ring_atoms = topology.get_ring_atoms(molecule)
        is_arom, num_arom = topology.is_aromatic(molecule) # AutoM3

        # Get Hbond information
        hbond_a = topology.get_hbond_a(feats)
        hbond_d = topology.get_hbond_d(feats)

        # Flatten list of ring atoms
        ring_atoms_flat = list(chain.from_iterable(ring_atoms))

        # Optimize coarse-grained bead positions -- keep all possibilities in case something goes
        # wrong later in the code.
        list_cg_beads, list_bead_pos = optimization.find_bead_pos(
            molecule,
            conf,
            list_heavy_atoms,
            self.heavy_atom_coords,
            ring_atoms,
            ring_atoms_flat, 
            force_map # AutoM3 new argument
        )

        # Loop through best 1% cg_beads and avg_pos
        max_attempts = int(math.ceil(0.5 * len(list_cg_beads)))
        logger.info(f"Max. number of attempts: {max_attempts}")
        attempt = 0

        while attempt < max_attempts:
            cg_beads = list_cg_beads[attempt]
            bead_pos = list_bead_pos[attempt]
            success = True

            # Remove mappings with bead numbers less than most optimal mapping.
            if (
                len(cg_beads) < len(list_cg_beads[0])
                and (len(list_heavy_atoms) - (5 * len(cg_beads))) > 3
            ):
                success = False

            # Extract position of coarse-grained beads
            cg_bead_coords = get_coords(conf, cg_beads, bead_pos, ring_atoms_flat)

            ### AutoM3 change : different partition of atoms into coarse-grained beads, depending on the number of aromatic cycles ###
            _, num_arom = topology.is_aromatic(molecule)

            if not force_map and num_arom<7: # AutoM3
                self.atom_partitioning,_ = optimization.voronoi_atoms_new( 
                    cg_bead_coords, self.heavy_atom_coords
                )
                _, self.cg_bead_coords = optimization.voronoi_atoms_new(
                    cg_bead_coords, self.atom_coords
                )
            else:
                self.atom_partitioning,_ = optimization.voronoi_atoms_old(
                    cg_bead_coords, self.heavy_atom_coords
                )
                _, self.cg_bead_coords = optimization.voronoi_atoms_old(
                    cg_bead_coords, self.atom_coords
                )
            
            
            # AutoM3 : trying mapping with at least 1 of 2 new conditions : 
            #    Max 2 aromatic atoms per bead ; 
            #    Holding Functional groups together in bead ;
            
            max_fails=1
            fails=0

            if is_arom and (num_arom % 2) == 0: #only for pair number of aromatic atoms (actual code prevents sharing/mismatch)
                if not optimization.max2arperbead(self.atom_partitioning, ring_atoms):
                    fails += 1

            if not optimization.functional_groups_ok(self.atom_partitioning,molecule, ring_atoms):
                fails += 1
            
            if force_map:
                if fails > max_fails:
                    success=False
                else:
                    success=True
            else:
                if fails>0: 
                    success=False


            logger.info("; Atom partitioning: {atom_partitioning}")

            self.cg_bead_names, bead_types, _, _ = topology.print_atoms(
                molname,
                forcepred,
                cg_beads,
                molecule,
                hbond_a,
                hbond_d,
                self.atom_partitioning,
                ring_atoms,
                ring_atoms_flat,
                logp_file, # AutoM3 new argument
                True,
            )

            if not self.cg_bead_names:
                success = False
            # Check additivity between fragments and entire molecule
            if not check_additivity(forcepred, bead_types, molecule, mol_smi):
                success = False
            
            # Bond list
            try:
                bond_list, const_list, _ = topology.print_bonds(
                    cg_beads,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    ring_atoms,
                    trial=True,
                )
            except Exception:
                raise

            # I added errval below from the master branch ... not sure where to use this anywhere, possibly leave for debugging
            if not ring_atoms and (len(bond_list) + len(const_list)) >= len(self.cg_bead_names):
                errval = 3
                success = False
            if (len(bond_list) + len(const_list)) < len(self.cg_bead_names) - 1:
                errval = 5
                success = False
            if len(cg_beads) != len(self.cg_bead_names):
                success = False
                errval = 8
            
            if success:
                header_write = topology.print_header(molname, mol_smi)
                self.cg_bead_names, bead_types, atoms_write, atoms_in_smi = topology.print_atoms( # AutoM3 new variable : atoms_in_smi
                    molname,
                    forcepred,
                    cg_beads,
                    molecule,
                    hbond_a,
                    hbond_d,
                    self.atom_partitioning,
                    ring_atoms,
                    ring_atoms_flat,
                    logp_file, # AutoM3 change
                    trial=False,
                )

                bond_list, const_list, bonds_write = topology.print_bonds(
                    cg_beads,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    ring_atoms,
                    False,
                )

                if not simple_model: # AutoM3
                    dihedrals_write = topology.print_dihedrals(
                    cg_beads,
                    const_list,
                    ring_atoms,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    molecule,
                    self.atom_partitioning
                    )

                angles_write, angle_list = topology.print_angles(
                    cg_beads,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    bond_list,
                    const_list,
                    ring_atoms,
                )

                if not angles_write and len(bond_list) > 1:
                    errval = 2
                if bond_list and angle_list:
                    if (len(bond_list) + len(const_list)) < 2 and len(angle_list) > 0:
                        errval = 6
                    if (
                        not ring_atoms
                        and (len(bond_list) + len(const_list)) - len(angle_list) != 1
                    ):
                        errval = 7


                self.topout, bartender_input_info = topology.topout(header_write,atoms_write,bonds_write,angles_write) # AutoM3 change : possible simple output w/o dihedrals, virtual sites


                ### AutoM3 outputs ###

                if len(ring_atoms)>0 and not simple_model:
                    if len(ring_atoms[0])>7:
                        vs_write, virtual_sites, vs_bead_coords  = topology.print_virtualsites(ring_atoms,cg_bead_coords,self.atom_partitioning,molecule)
                        
                        self.topout, vs_bead_names, bartender_input_info  = topology.topout_vs(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, virtual_sites,vs_write,simple_model)
                    
                    else:
                        self.topout, bartender_input_info = topology.topout_noVS(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, self.cg_bead_coords, ring_atoms, cg_beads)
                
                if bartender:
                    bartender_out = topology.bartender_input(molecule, molname, atoms_in_smi, bartender_input_info)
                    with open(bartenderfname, "w") as btf:
                        btf.write(bartender_out)
                
                if topfname:
                    with open(topfname, "w") as fp:
                        fp.write(self.topout)
                if not force_map: print("Converged to solution in {} iteration(s)".format(attempt + 1))
                if force_map: print("Converged to solution in {} iteration(s)".format(attempt + 1 + max_attempts))
                break
            else:
                attempt += 1
        
                # AutoM3 change : force mapping by old code if new code doesn't give result
                if attempt == max_attempts and not force_map:
                    force_map=True
                    attempt = 0 

        if attempt == max_attempts and force_map:
            raise RuntimeError(
                "ERROR: no successful mapping found.\nTry running with the --fpred and/or --verbose options."
            )
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
