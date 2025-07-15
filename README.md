Auto_MartiniM3
============

## What is Auto_MartiniM3?

A toolkit that enables automatic generation of Martini force field for small organic molecules up to 25 heavy atoms, now in agreement with Martini 3 Force Field parameters. 

## Developers
* Magdalena Szczuka (University Toulouse 3, France)
* Tristan Bereau (University of Amsterdam, Netherlands)   
* Kiran Kanekal (Max Planck Institute for Polymer Research, Mainz, Germany)     
* Andrew Abi-Mansour (Molecular Sciences Software Institute, Virginia Tech, Blacksburg, US)

## Supervisors
* Matthieu Chavent (Centre de Biologie Intégrative (CBI)), University Toulouse 3, CNRS, France)
* Pierre Poulain (Université Paris Cité, France)
* Paulo C. T. Souza (Laboratoire de Biologie et Modélisation de la Cellule, UMR 5239, ENS Lyon, France)

## Installation with conda

 For enabling automatic mapping with `Auto-MartiniM3`, you need to clone this repository and create a conda environment.

```bash
git clone https://github.com/Martini-Force-Field-Initiative/Automartini_M3.git
cd Automartini_M3
conda env create -f environment.yaml
```

This will create a conda environment called `automartiniM3` which you can activate with

```bash
conda activate automartiniM3
```

## Testing

To run the test cases and validate your installation, you will need to have [pytest](https://docs.pytest.org/en/stable/getting-started.html) 
installed. If you installed `auto_martiniM3` with conda, then pytest should already be available in your environment.

To initiate the testing, run the following:
```bash
pytest -v tests
```

All tests should pass within few minutes. If any of the tests fail, please open an [issue](https://github.com/Martini-Force-Field-Initiative/Automartini_M3/issues).

## Command-line Interface
You can invoke `auto_martiniM3` from the command-line via:
```
python -m auto_martiniM3 [mode] [options]
```
By default, mode is set to 'run', which computes the MARTINI 3 force field for a given molecule.

To display the usage-information (help), either supply -h, --help, or nothing to auto_martiniM3:
 
```
usage: auto_martiniM3 [-h] [--mode {run}] [--sdf SDF | --smi SMI] [--logp LOGP] [--mol MOLNAME] [--aa AA] [-v]
                      [--fpred] [--bartender] [--simple] [--canon]

Generates Martini 3 force field for atomistic structures of small organic molecules

optional arguments:
  -h, --help     show this help message and exit
  --mode {run}   mode: run (compute FF)
  --sdf SDF      SDF file of atomistic coordinates
  --smi SMI      SMILES string of atomistic structure
  --logp LOGP    File with partial smiles and associated logP
  --mol MOLNAME  Name of CG molecule
  --aa AA        filename of all-atom structure .gro file
  -v, --verbose  increase verbosity
  --fpred        Atomic partitioning prediction
  --bartender    Bartender input file
  --simple       Simple model without dihedrals nor virtual sites
  --canon        Translate to RdKit canon structure

Developers:
===========
Magdalena Szczuka (magdalena.szczuka [at] univ-tlse3.fr)
Tristan Bereau (bereau [at] mpip-mainz.mpg.de)
Kiran Kanekal (kanekal [at] mpip-mainz.mpg.de)
Andrew Abi-Mansour (andrew.gaam [at] gmail.com)
```

## Example
To coarse-grain a molecule, simply provide its SMILES code (option `--smi SMI`) or a .SDF file (option `'--sdf file.sdf`). You also need to provide a name for the CG molecule (not longer than 5 characters) using the `--mol` option.  For instance, to coarse grain [aspirin](https://pubchem.ncbi.nlm.nih.gov/compound/2244#section=2D-Structure), you can either obtain/generate (e.g., from Open Babel) an SDF file:
```
python -m auto_martiniM3 --sdf aspirin.sdf --mol ASP 
```
(the name ASP is arbitrary) or use its SMILES code within double quotes
```
python -m auto_martiniM3 --smi "CC(=O)OC1=CC=CC=C1C(=O)O" --mol ASP 
```
In case no problem arises, it will output the gromacs ASP.itp file:
```
; GENERATED WITH Auto_Martini M3FF for ASP
; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour
; updated to Martini 3 force field by Magdalena Szczuka
; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza 
; SMILES code : CC(=O)OC1=CC=CC=C1C(=O)O


[moleculetype]
; molname       nrexcl
  ASP          1

[atoms]
; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num
   1       SN5a    1   ASP     N01       1        0    54   ;   CC=O     ; atoms: C0, C1, O2,          
   2       TP2a    1   ASP     P01       2        0    36   ;   CO       ; atoms: O3, C4,          
   3       TC5     1   ASP     C01       3        0    36   ;   C=C      ; atoms: C5, C6,          
   4       SC5     1   ASP     C02       4        0    54   ;   CC=C     ; atoms: C7, C8, C9,          
   5       SN6d    1   ASP     N02       5        0    54   ;   O=CO     ; atoms: C10, O11, O12, ; ALOGPS defined bead


[bonds]
;  i   j     funct   length   force.c.
   1   2     1       0.27       25000.00
   2   3     1       0.27       25000.00
   2   4     1       0.33       10000.00
   3   4     1       0.28       100000.00
   4   5     1       0.35       5000.00
#ifndef FLEXIBLE
[constraints]
#endif
;  i   j     funct   length

[angles]
;  i  j  k    funct  angle  force.c.
   1  2  5       1    95.8   100.0
   1  4  5       1    52.7    25.0

[dihedrals]
;  i  j  k  l  funct  angle  force.c.
   1  2  3  4    2    -135.5   25.0   
   2  3  4  5    2    -0.5     25.0   

[exclusions]
  1 4
```
The code will also output a corresponding `.gro` file for the coarse-grained coordinates.
Atomistic coordinates can be written using the `--aa output.gro` option.

If `--bartender` flag is used, additional file for further optimization of bonded parameters with Bartender [Pereira et al., 2024](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00275) will be produced. You will find more information about Bartender in the [official tutorial](https://github.com/Martini-Force-Field-Initiative/Bartender.git). Examplary Bartender input file, created by Auto-MartiniM3 for aspirin and saved as ASP_bartender.inp :
```
# INPUT data for bonded parameter definition by BARTENDER for molecule ASP
BEADS
1 1,2,3,14,15,16
2 4,5
3 6,7,17,18
4 8,9,10,19,20
5 11,12,13,21
BONDS
1,2
2,3
2,4
3,4
4,5
ANGLES
1,2,5
1,4,5
IMPROPERS
1,2,3,4
2,3,4,5
```
## Caveats

For frequently encountered problems, see [FEP](FEP.md).

