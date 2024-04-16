Auto_Martini M3
============

## What is Auto_Martini?
A toolkit that enables automatic generation of Martini forcefields for small organic molecules, now in agreement with Martini 3 Force Field parameters. 

For a detailed account of the software, see:

still writing...

## Developers 
* Tristan Bereau (University of Amsterdam, Netherlands)   
* Kiran Kanekal (Max Planck Institute for Polymer Research, Mainz, Germany)     
* Andrew Abi-Mansour (Molecular Sciences Software Institute, Virginia Tech, Blacksburg, US)
* Magdalena Szczuka

## Installation with conda
 For enabling automatic mapping with `Auto-Martini`, you need to create conda environment.
```bash
cd auto_martini
conda env create -f environment.yaml
```
This will create a conda environment called `autom` which you can activate with
```bash
conda activate autom
```
Now use pip from the `auto_martini` src dir to run the installation:
```bash
pip install .
```

## Testing
To run the test cases and validate your installation, you will need to have [pytest](https://docs.pytest.org/en/stable/getting-started.html) 
installed. If you installed `auto_martini` with conda, then pytest should already be available in your environment.

To initiate the testing, run the following:
```bash
pytest -v tests
```

All tests should pass within few minutes. If any of the tests fail, please open an [issue](https://github.com/tbereau/auto_martini/issues).

## Command-line Interface
You can invoke `auto_martini` from the command-line via:
```
python -m auto_martini [mode] [options]
```
By default, mode is set to 'run', which computes the MARTINI forcefield for a given molecule.

To display the usage-information (help), either supply -h, --help, or nothing to auto_martini:
 
```
usage: auto_martini [-h] [--mode {run,test}] [--sdf SDF | --smi SMI]
                    [--mol MOLNAME] [--aa AA] [-v] [--fpred] [--dih True(default) / False]

Generates Martini force field for atomistic structures of small organic molecules

optional arguments:
  -h, --help         show this help message and exit
  --mode {run,test}  mode: run (compute FF) or test (validate)
  --sdf SDF          SDF file of atomistic coordinates
  --smi SMI          SMILES string of atomistic structure
  --mol MOLNAME      Name of CG molecule
  --aa AA            filename of all-atom structure .gro file
  -v, --verbose      increase verbosity
  --fpred            Atomic partitioning prediction
  --dih		     False, if simplified mapping wanted (without dihedrals nor virtual sites)

Developers:
===========
Tristan Bereau (bereau [at] mpip-mainz.mpg.de)
Kiran Kanekal (kanekal [at] mpip-mainz.mpg.de)
Andrew Abi-Mansour (andrew.gaam [at] gmail.com)
Magdalena Szczuka (magdalena.szczuka [at] univ-tlse3.fr)
```

## Example
To coarse-grain a molecule, simply provide its SMILES code (option `--smi SMI`) or a .SDF file (option `'--sdf file.sdf`). You also need to provide a name for the CG molecule (not longer than 5 characters) using the `--mol` option.  For instance, to coarse grain [aspirin](https://pubchem.ncbi.nlm.nih.gov/compound/2244#section=2D-Structure), you can either obtain/generate (e.g., from Open Babel) an SDF file:
```
python -m auto_martini --sdf aspirin.sdf --mol ASP 
```
(the name GUA is arbitrary) or use its SMILES code within double quotes
```
python -m auto_martini --smi "N1=C(N)NN=C1N" --mol ASP 
```
In case no problem arises, it will output the gromacs GUA.itp file:
```
; GENERATED WITH auto_Martini v0.0.0 for ASP
; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour
; updated to Martini3 by Magdalena Szczuka, reviewed by Matthieu Chavent 
; SMILE code : CC(=O)OC1=CC=CC=C1C(=O)O

[moleculetype]
; molname       nrexcl
  ASP          1

[atoms]
; id    type    resnr   residue  atom    cgnr    charge    mass  smiles
   1       SN3a    1   ASP     N01       1        0    40   ;   CC=O
   2       TP3a    1   ASP     P01       2        0    28   ;   CO
   3       SC2     1   ASP     C01       3        0    36   ;   CC=C
   4       TC5     1   ASP     C02       4        0    24   ;   C=C
   5       SN6d    1   ASP     N02       5        0    44   ;   OC=O

[bonds]
;  i   j     funct   length   force.c.
   1   2     1       0.29       5000.00
   4   5     1       0.26       5000.00

#ifndef FLEXIBLE
[constraints]
#endif
;  i   j     funct   length
   1   3     1       0.54    1000000
   1   5     1       0.56    1000000
   2   3     1       0.30    1000000
   2   4     1       0.30    1000000
   2   5     1       0.34    1000000
   3   4     1       0.27    1000000
   3   5     1       0.50    1000000

[angles]
; i j k         funct   angle   force.c.
  1 2 5         2       123.7   25.0
  1 4 5         2       71.7   100.0

[dihedrals]
;  i j k l   funct   angle  force.c.
  1 2 3 4       2     167.2  100.0
  2 3 4 5       2     1.0    100.0

[exclusions]
  1 4

```
The code will also output a corresponding `.gro` file for the coarse-grained coordinates
Atomistic coordinates can be written using the `--aa output.gro` option.

## Caveats

For frequently encountered problems, see [FEP](FEP.md).

