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

from .common import *

logger = logging.getLogger(__name__)

from sys import exit


def draw_molecule_with_colored_groups(mol, atom_groups, groupnames=None, output_filename="mapping.png"):
    """
    Draw a molecule and overlay semi-transparent circles around atom groups.

    mol : RDKit Mol
    atom_groups : list of sets of atom indices
    groupnames : list of group type strings for color mapping
    """
    if mol is None:
        raise ValueError("Invalid MOL object")

    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(mol)

    # Draw molecule normally (no atom highlighting)
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", atom.GetSymbol())
    width, height = 500, 500
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Convert RDKit drawing to PIL Image
    image_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(image_data)).convert("RGBA")

    # Make room for the filename label below
    padding = 120
    new_img = Image.new("RGBA", (img.width, img.height + padding), (255, 255, 255, 255))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    # Load font
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"
        font = ImageFont.truetype(font_path, size=40)
        small_font = ImageFont.truetype(font_path, size=22)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default(22)

    # Filename label
    name_only = os.path.splitext(os.path.basename(output_filename))[0]
    bbox = draw.textbbox((0, 0), name_only, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img.width - text_width) // 2
    text_y = img.height + (padding - text_height) // 100
    draw.text((text_x, text_y), name_only, fill=(0, 0, 0), font=font)

    # Get 2D coordinate mapping
    conf = mol.GetConformer()
    raw_coords = {i: conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())}
    draw_coords = {i: drawer.GetDrawCoords(i) for i in range(mol.GetNumAtoms())}

    # --- COLOR MAPPING FROM groupnames ---
    color_map = {
        "C": (212, 175, 55, 70),   # gold
        "N": (30, 90, 255, 70),    # blue
        "P": (30, 200, 30, 70),    # green
        "X": (230, 0, 0, 70),      # red
        "Q": (145, 0, 200, 70),    # purple
    }

    # Fallback color if groupname missing
    default_color = (150, 150, 150, 120)

    # --- DRAW SEMI-TRANSPARENT CIRCLES AROUND GROUPS ---
    overlay = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for idx, group in enumerate(atom_groups):
        if not group:
            continue

        # Determine circle color from groupname
        if groupnames and idx < len(groupnames):
            groupcode = None
            for letter in ["C", "N", "P", "X", "Q"]:
                if letter in groupnames[idx]:
                    groupcode = letter
                    break
            color = color_map.get(groupcode, default_color)
        else:
            color = default_color

        # Extract 2D pixel coordinates of atoms
        pts = []
        for a in group:
            pt = draw_coords[a]
            pts.append((float(pt.x), float(pt.y)))

        # Compute center of geometry
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # Maximum distance to any atom
        radius = max(math.dist((cx, cy), p) for p in pts)

        # Expand radius slightly for visual padding
        radius *= 1.25

        # Draw circle (ellipse)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        overlay_draw.ellipse(bbox, fill=color, outline=None)

        # Label group
        label = groupnames[idx] if groupnames else f"#{idx+1}"
        overlay_draw.text((cx, cy), label, fill=color[:3], font=small_font)

    # Blend circles with image
    result = Image.alpha_composite(new_img, overlay)
    result.save(output_filename)

    print(f"✅ Saved image to {output_filename} with circular group overlays")



def draw_molecule_with_colored_groups_highlights(mol, atom_groups, groupnames=None, output_filename="mapping.png"):
    """
    INPUTS:
    mol = molecule RdKit object
    atom_groups = list of sets : [ {set1}, ... ]
    groupnames = list of bead types in the same order as corresponding "group"
    """
    if mol is None:
        raise ValueError("Invalid MOL object")
    
    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(mol)

    mol=Chem.RemoveHs(mol)

    # Assign random but visually distinct colors to each group
    colors = {
        i: (
            random.random() * 0.6 + 0.4,
            random.random() * 0.6 + 0.3,
            random.random() * 0.6 + 0.4
        )
        for i in range(len(atom_groups))
    }

    # Prepare atom highlights
    atom_colors = {}
    highlight_atoms = set()
    for idx, group in enumerate(atom_groups):
        for atom_idx in group:
            atom_colors[atom_idx] = colors[idx]
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() != "H": highlight_atoms.add(atom_idx)

    # Create drawer and draw molecule
    width, height = 1000, 1000
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addAtomIndices = True  # Show atom indices
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms),
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    image_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(image_data)).convert("RGBA")

    # Make room for label text below
    padding = 120
    new_img = Image.new("RGBA", (img.width, img.height + padding), (255, 255, 255, 255))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    # Try to load a large TTF font
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"
        font = ImageFont.truetype(font_path, size=40)
        small_font = ImageFont.truetype(font_path, size=28)
    except Exception:
        print("⚠️ Using fallback font")
        font = ImageFont.load_default(40)
        small_font = ImageFont.load_default(28)

    # Add filename label below
    name_only = os.path.splitext(os.path.basename(output_filename))[0]
    bbox = draw.textbbox((0, 0), name_only, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img.width - text_width) // 2
    text_y = img.height + (padding - text_height) // 2
    draw.text((text_x, text_y), name_only, fill=(0, 0, 0), font=font)

    # Get 2D coordinates for group label placement
    conf = mol.GetConformer()
    coord_map = {atom.GetIdx(): conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()}
    # Scale factor based on image size
    scale = drawer.GetDrawCoords(0).x / coord_map[0].x if coord_map[0].x != 0 else 1.0

    # Add group number labels near first atom of each group
    for idx, group in enumerate(atom_groups):
        if not group:
            continue
        atom_idx = next(iter(group))
        pt = drawer.GetDrawCoords(atom_idx)  # Get 2D coordinates in pixel space
        label = f"#{idx+1}"
        if groupnames: label = groupnames[idx]
        color_rgb = tuple(int(c * 255) for c in colors[idx])
        label_x = int(pt.x) + 1
        label_y = int(pt.y) - 100
        draw.text((label_x, label_y), label, fill=color_rgb, font=font)

    new_img.save(output_filename)
    print(f"✅ Saved image to {output_filename} with group labels")


def output_gro(sites, site_names, molname, molecule=None, mapping=None, bead_types=None):
    """Output GRO file of CG structure"""
    logger.debug("Entering output_gro()")
    num_beads = len(sites)
    gro_out = ""
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    gro_out += "{:s} generated from auto_martiniM3 beta\n".format(molname)
    gro_out += "{:5d}\n".format(num_beads)

    for i in range(num_beads):
        gro_out += "{:5d}{:<6s}{:4s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
            1, #was i +1, but this is GRO file for one molecule, so all beads should be a part of the same molecule
            molname[:5],
            site_names[i],
            i + 1,
            sites[i][0] / 10.0,
            sites[i][1] / 10.0,
            sites[i][2] / 10.0,
        )
    gro_out += "{:10.5f}{:10.5f}{:10.5f}\n".format(10.0, 10.0, 10.0)
    if molecule: 
        draw_molecule_with_colored_groups(molecule, mapping, bead_types, f"{molname}_CGmapping.png")
        draw_molecule_with_colored_groups_highlights(molecule, mapping, bead_types, f"{molname}_CGhighlights.png")
    return gro_out
