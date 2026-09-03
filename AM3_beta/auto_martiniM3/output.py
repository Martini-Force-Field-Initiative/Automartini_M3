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
import cv2

logger = logging.getLogger(__name__)

from sys import exit

_STATE_RANGES = {
    "C": {"min": 1, "max": 6, "c1": (255, 215, 0), "c2": (255, 69, 0)},   # gold -> orangered
    "N": {"min": 1, "max": 6, "c1": (0, 255, 255), "c2": (138, 43, 226)}, # cyan -> blueviolet
    "P": {"min": 1, "max": 6, "c1": (173, 255, 47), "c2": (0, 100, 0)},   # greenyellow -> darkgreen
    "X": {"min": 1, "max": 4, "c1": (160, 32, 240), "c2": (255, 0, 0)},   # purple -> red
}
_PINK_C1 = (255, 182, 193)  # lightpink
_PINK_C2 = (255, 20, 147)   # deeppink


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _get_state_color(name):
    """Return an (r,g,b) 0-255 tuple derived from a group name."""
    if name:
        for letter in ("C", "N", "P", "X"):
            m = re.search(rf"{letter}(\d+)", name)
            if m:
                n = int(m.group(1))
                rng = _STATE_RANGES[letter]
                n_clamped = max(rng["min"], min(rng["max"], n))
                t = (n_clamped - rng["min"]) / (rng["max"] - rng["min"])
                return _lerp_color(rng["c1"], rng["c2"], t)
    # fallback: pink family, spread deterministically by name hash
    t = (hash(name) % 1000) / 1000.0 if name else random.random()
    return _lerp_color(_PINK_C1, _PINK_C2, t)

def _boxes_intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _occupancy_fraction(occupied_mask, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = occupied_mask.shape
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 1.0  # out of bounds / degenerate -> treat as fully bad
    region = occupied_mask[y0:y1, x0:x1]
    if region.size == 0:
        return 1.0
    return float(region.mean())


def _find_label_position(draw, font, text, anchor_xy, occupied_mask,
                          placed_boxes, canvas_size,
                          max_occupancy=0.04, max_radius=260,
                          radius_step=22, min_radius=34):
    """
    Search outward from anchor_xy for a text box location that mostly
    avoids occupied pixels and does not overlap previously placed labels.
    Returns (x0, y0) top-left corner for drawing the text, and whether
    the label had to be displaced from the anchor.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax, ay = anchor_xy
    cw, ch = canvas_size

    angles = [0, 45, -45, 90, -90, 135, -135, 180]  # right, diag, up/down, left first
    angles = [math.radians(a) for a in angles]

    best_box, best_score = None, None

    radius = min_radius
    while radius <= max_radius:
        for ang in angles:
            cx = ax + radius * math.cos(ang)
            cy = ay + radius * math.sin(ang)
            x0, y0 = cx - tw / 2, cy - th / 2
            x1, y1 = x0 + tw, y0 + th
            # keep fully inside canvas
            if x0 < 2 or y0 < 2 or x1 > cw - 2 or y1 > ch - 2:
                continue
            box = (x0, y0, x1, y1)
            occ = _occupancy_fraction(occupied_mask, box)
            overlaps_placed = any(_boxes_intersect(box, pb) for pb in placed_boxes)
            score = occ + (1.0 if overlaps_placed else 0.0)
            if best_score is None or score < best_score:
                best_score, best_box = score, box
            if occ <= max_occupancy and not overlaps_placed:
                return (x0, y0), box, True
        radius += radius_step

    # fallback: best effort location found during search
    if best_box is not None:
        return (best_box[0], best_box[1]), best_box, True
    # absolute fallback: draw right at the anchor
    x0, y0 = ax - tw / 2, ay - th / 2
    return (x0, y0), (x0, y0, x0 + tw, y0 + th), False

def _rgb255_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _median_bond_pixel_length(mol, coords_lookup):
    lengths = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in coords_lookup and a2 in coords_lookup:
            p1, p2 = coords_lookup[a1], coords_lookup[a2]
            lengths.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return float(np.median(lengths)) if lengths else 40.0

def _group_contours(mol, atom_groups, coords_lookup, width, height,
                     radius_frac=0.27, bond_width_frac=.80, simplify_eps=1.5):

    median_bond = _median_bond_pixel_length(mol, coords_lookup)
    radius = radius_frac * median_bond
    bond_width = max(2, int(bond_width_frac * radius * 2))

    all_contours = {}
    for idx, group in enumerate(atom_groups):
        if not group:
            all_contours[idx] = []
            continue
        heavy = [a for a in group if a in coords_lookup]
        heavy_set = set(heavy)

        mask_img = Image.new("L", (width, height), 0)
        d = ImageDraw.Draw(mask_img)
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in heavy_set and a2 in heavy_set:
                d.line([coords_lookup[a1], coords_lookup[a2]], fill=255, width=bond_width)
        for a in heavy:
            p = coords_lookup[a]
            d.ellipse([p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius], fill=255)

        mask_arr = np.array(mask_img)
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified = []
        for c in contours:
            if len(c) < 3:
                continue
            approx = cv2.approxPolyDP(c, simplify_eps, True)
            pts = approx.reshape(-1, 2)
            if len(pts) >= 3:
                simplified.append(pts)
        all_contours[idx] = simplified
    return all_contours




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
    mol=Chem.RemoveHs(mol)
    Chem.rdDepictor.Compute2DCoords(mol)

    # Draw molecule normally (no atom highlighting)
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", atom.GetSymbol())
    width, height = 1000, 1000
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



def draw_molecule_with_colored_groups_highlights(
        mol, atom_groups, groupnames=None, 
        output_filename="mapping.png",statecolors=False,fmt="svg",
        group_border=True, border_color=(102, 102, 102), border_width=1,
):
    """
    INPUTS:
    mol = molecule RdKit object
    atom_groups = list of sets : [ {set1}, ... ]
    groupnames = list of bead types in the same order as corresponding "group"
    statecolors = if True, colors are derived deterministically from groupnames
    group_border = if True (default), draw a thin outline around each
                  group's combined highlighted region
    border_color = (r, g, b) 0-255 color used for the group outlines.
    border_width = outline stroke width in pixels.
    """
    if mol is None:
        raise ValueError("Invalid MOL object")
    
    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(mol)
    mol=Chem.RemoveHs(mol)

    #Asign highlight colors
    colors_float = {}   # 0-1 floats, used by RDKit
    colors_rgb255 = {}  # 0-255 ints, used for text/leader lines

    for i in range(len(atom_groups)):
        if statecolors:
            name = groupnames[i] if groupnames and i < len(groupnames) else None
            rgb255 = _get_state_color(name)
        else: #random colors
            rgb255 = tuple(int(255 * (random.random() * 0.6 + (0.4 if k == 0 else 0.3 if k == 1 else 0.4)))
                            for k in range(3))
        colors_rgb255[i] = rgb255
        colors_float[i] = tuple(c / 255.0 for c in rgb255)

    colors = colors_float

    # Prepare atom highlights
    atom_colors = {}
    highlight_atoms = set()
    for idx, group in enumerate(atom_groups):
        for atom_idx in group:
            atom_colors[atom_idx] = colors[idx]
            if mol.GetAtomWithIdx(atom_idx).GetSymbol() != "H": highlight_atoms.add(atom_idx)

    # Create drawer and draw molecule
    width, height = 1000, 1000
    padding = 120

    cairo_drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    cairo_drawer.drawOptions().addAtomIndices = True
    cairo_drawer.DrawMolecule(
        mol, highlightAtoms=list(highlight_atoms), highlightAtomColors=atom_colors
    )
    cairo_drawer.FinishDrawing()
    probe_img = Image.open(io.BytesIO(cairo_drawer.GetDrawingText())).convert("RGB")
    arr = np.array(probe_img)

    # coordinate lookup used both for contour-building and label anchors
    coords_lookup = {a.GetIdx(): (cairo_drawer.GetDrawCoords(a.GetIdx()).x,
                                    cairo_drawer.GetDrawCoords(a.GetIdx()).y)
                        for a in mol.GetAtoms()}

    group_contours = {}
    if group_border:
        group_contours = _group_contours(mol, atom_groups, coords_lookup, width, height)
        # burn the outlines directly into the raster array so they are
        # present both in the final PNG and in the occupancy mask used
        # for label placement (labels will avoid the border lines too)
        for idx, contours in group_contours.items():
            for c in contours:
                cv2.polylines(
                    arr, [c.reshape(-1, 1, 2)], isClosed=True,
                    color=border_color, thickness=border_width, lineType=cv2.LINE_AA
                )

    occ_mol = np.any(arr < 250, axis=-1)  # True where non-background
    occupied_mask = np.zeros((height + padding, width), dtype=bool)
    occupied_mask[:height, :width] = occ_mol
    

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

    # dummy PIL draw context, used only to measure text bounding boxes
    dummy_img = Image.new("RGB", (width, height + padding), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy_img)

    placed_boxes = []
    canvas_size = (width, height + padding)

    # Reserve the filename box first so group labels avoid it
    name_only = os.path.splitext(os.path.basename(output_filename))[0]
    bbox = dummy_draw.textbbox((0, 0), name_only, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width) // 2
    text_y = height + (padding - text_height) // 2
    placed_boxes.append((text_x, text_y, text_x + text_width, text_y + text_height))

    # Work out label positions (shared between PNG and SVG paths)
    label_placements = []  # list of dicts: label, color, box, anchor, lx, ly
    for idx, group in enumerate(atom_groups):
        if not group:
            continue
        heavy_atoms = [a for a in group if a in coords_lookup] or list(group)
        pts = [coords_lookup[a] for a in heavy_atoms]
        anchor_x = sum(p[0] for p in pts) / len(pts)
        anchor_y = sum(p[1] for p in pts) / len(pts)

        label = f"#{idx+1}"
        if groupnames and idx < len(groupnames):
            label = groupnames[idx]
        color_rgb = colors_rgb255[idx]

        (lx, ly), box, _ = _find_label_position(
            dummy_draw, font, label, (anchor_x, anchor_y),
            occupied_mask, placed_boxes, canvas_size
        )

        placed_boxes.append(box)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(canvas_size[0], x1); y1 = min(canvas_size[1], y1)
        if x1 > x0 and y1 > y0:
            occupied_mask[y0:y1, x0:x1] = True

        label_placements.append(dict(
            label=label, color_rgb=color_rgb, lx=lx, ly=ly,
        ))


    if fmt == "png":
        img = Image.fromarray(arr, mode="RGB").convert("RGBA")
        new_img = Image.new("RGBA", (img.width, img.height + padding), (255, 255, 255, 255))
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        draw.text((text_x, text_y), name_only, fill=(0, 0, 0), font=font)
        for lp in label_placements:
            draw.text((lp["lx"], lp["ly"]), lp["label"], fill=lp["color_rgb"], font=font)

        new_img.save(output_filename)
        print(f"✅ Saved PNG image to {output_filename} with group labels")

    else:  # fmt == "svg"
        svg_drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        svg_drawer.drawOptions().addAtomIndices = True
        svg_drawer.DrawMolecule(
            mol, highlightAtoms=list(highlight_atoms), highlightAtomColors=atom_colors
        )
        svg_drawer.FinishDrawing()
        svg_text = svg_drawer.GetDrawingText()


        extra_svg = []
        font_family = "Arial, Helvetica, sans-serif"

        extra_svg.append(
            f'<text x="{width/2:.1f}" y="{text_y + text_height:.1f}" '
            f'font-family="{font_family}" font-weight="bold" font-size="40" '
            f'text-anchor="middle" fill="#000000">{name_only}</text>'
        )

        if group_border:
            border_hex = _rgb255_to_hex(border_color)
            for idx, contours in group_contours.items():
                for c in contours:
                    pts_str = " ".join(f"{x},{y}" for x, y in c)
                    extra_svg.append(
                        f'<polygon points="{pts_str}" fill="none" '
                        f'stroke="{border_hex}" stroke-width="{border_width}" '
                        f'stroke-linejoin="round"/>'
                    )

        for lp in label_placements:
            hexcolor = _rgb255_to_hex(lp["color_rgb"])
            # baseline offset (textbbox top-left -> SVG uses baseline for y)
            baseline_y = lp["ly"] + font.size
            extra_svg.append(
                f'<text x="{lp["lx"]:.1f}" y="{baseline_y:.1f}" '
                f'font-family="{font_family}" font-weight="bold" font-size="{font.size}" '
                f'fill="{hexcolor}">{lp["label"]}</text>'
            )

        # Widen the SVG canvas
        for q in ("'", '"'):
            svg_text = svg_text.replace(
                f'height={q}{height}px{q}', f'height={q}{height + padding}px{q}'
            )
        svg_text = re.sub(
            r'''viewBox=(['"])0 0 (\d+) (\d+)\1''',
            lambda m: f'viewBox={m.group(1)}0 0 {m.group(2)} {height + padding}{m.group(1)}',
            svg_text,
        )
        svg_text = svg_text.replace(
            "</svg>", "\n".join(extra_svg) + "\n</svg>"
        )

        with open(output_filename, "w") as f:
            f.write(svg_text)
        print(f"✅ Saved SVG image to {output_filename} with group labels")
    


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
        draw_molecule_with_colored_groups_highlights(molecule, mapping, bead_types, f"{molname}.png",statecolors=True,fmt="png")
    return gro_out
