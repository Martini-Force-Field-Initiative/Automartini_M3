"""
Created on August 7, 2021 by Andrew Abi-Mansour

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

AUTO_MARTINI is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
"""

import numpy
from Cython.Build import cythonize


def build(setup_kwargs):
    setup_kwargs.update(
        ext_modules=cythonize(["auto_martini/optimization.pyx"]),
        include_dirs=numpy.get_include(),
    )
