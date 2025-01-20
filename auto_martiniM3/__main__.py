"""
Created on March 17, 2019 by Andrew Abi-Mansour
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

import argparse
import logging
import sys

from . import __version__, solver
from .common import *
from .topology import gen_molecule_sdf, gen_molecule_smi

def checkArgs(args):
    """ AutoM3 change: removed argument --top for simpler input, .itp file will be named by using --mol argument (mol.itp)"""

    if not args.sdf and not args.smi:
        parser.error("run requires --sdf or --smi")

    if not args.molname:
        parser.error("run requires --mol")

parser = argparse.ArgumentParser(
    prog="auto_martiniM3",
    description="Generates Martini 3 force field for atomistic structures of small organic molecules",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Developers:\n===========\nMagdalena Szczuka (magdalena.szczuka [at] univ-tlse3.fr)\nTristan Bereau (bereau [at] mpip-mainz.mpg.de)\nKiran Kanekal (kanekal [at] mpip-mainz.mpg.de)
Andrew Abi-Mansour (andrew.gaam [at] gmail.com)""",
)
parser.add_argument(
    "--mode", type=str, choices=["run"], default="run", help="mode: run (compute FF)"
)

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument(
    "--sdf",
    dest="sdf",
    type=str,
    required=False,
    help="SDF file of atomistic coordinates",
)

group.add_argument(
    "--smi",
    dest="smi",
    type=str,
    required=False,
    help="SMILES string of atomistic structure",
)
parser.add_argument("--logp",dest="logp",type=str,required=None,help="File with partial smiles and associated logP") #AutoM3 change: for custom database of fragments and logp, default file in scripts directory (logP_smi.dat)
parser.add_argument("--mol", dest="molname", type=str, required=False, help="Name of CG molecule")
parser.add_argument("--aa", dest="aa", type=str, help="filename of all-atom structure .gro file")
parser.add_argument("-v","--verbose",dest="verbose",action="count",default=0,help="increase verbosity",)
parser.add_argument("--fpred",dest="forcepred",action="store_true",help="Atomic partitioning prediction",)
parser.add_argument("--bartender",dest="bartender_output",type=str,default=False,help="True, for generating bartender input file") #AutoM3 change
parser.add_argument("--simple",dest="simple_model",type=str,default=False, help="True, for simple model without dihedrals nor virtual sites in topology file.") #AutoM3 change

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

checkArgs(args)

if args.verbose >= 2:
    level = logging.DEBUG
elif args.verbose >= 1:
    level = logging.INFO
else:
    level = logging.WARNING

logging.basicConfig(
    filename="auto_martiniM3.log",
    format="%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s",
    level=level,
)

logger = logging.getLogger(__name__)

logger.info("Running auto_martiniM3 v{}".format(__version__))

# Generate molecule's structure from SDF or SMILES
if args.sdf:
    mol = gen_molecule_sdf(args.sdf)
    smiles = str(Chem.MolToSmiles(mol, isomericSmiles=False))
    #mol, _ = gen_molecule_smi(smiles)
else:
    mol, _ = gen_molecule_smi(args.smi)
    smiles = args.smi

### AutoM3 change ###
topname=args.molname+".itp"
groname=args.molname+".gro"
bartenderfname=""

if args.bartender_output:
    bartenderfname=args.molname+"_bartenderINPUT.dat"
    cg = solver.Cg_molecule(mol, smiles, args.molname, args.simple_model, topname, bartenderfname, args.bartender_output, args.logp, args.forcepred)
else:
    cg = solver.Cg_molecule(mol, smiles, args.molname, args.simple_model, topname, bartenderfname, args.bartender_output, args.logp, args.forcepred)
if args.aa:
    cg.output_aa(args.aa)
cg.output_cg(groname)