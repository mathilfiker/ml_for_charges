"""
- Draws the heavy-atom-only 2D sketch with forced numeric labels on each heavy atom.
- Draws a grouped bar plot of per-atom variability (std) for each method.
- Bar facecolor = mean charge; bar height = std.

"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem import RWMol
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------
# USER PARAMETERS
# -----------------------
molecule = "mobley_9897248"   # base name for "<molecule>.sdf"
sdf_path = f"conformers/{molecule}.sdf"
methods = ["BP", "ESP", "RESP"]
conf_glob_template = os.path.join("charges/{method}", f"{molecule}_conf_*.npy")
out_filename = f"{molecule}_charge_variability.png"
figsize = (12, 5) #it was 12
dpi = 300
cmap_name = "seismic"
# -----------------------

# ---- load molecule (first record) ----
supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
mol0 = next((m for m in supplier if m is not None), None)
if mol0 is None:
    raise FileNotFoundError(f"No valid molecule found in {sdf_path}")

# ---- build heavy-atom-only mol and mappings ----
orig_atoms = list(mol0.GetAtoms())
orig_idx_to_new_idx = {}
new_idx_to_orig_idx = {}
rw = RWMol()
for a in orig_atoms:
    if a.GetAtomicNum() == 1:
        continue
    new_idx = rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
    orig_idx_to_new_idx[a.GetIdx()] = new_idx
    new_idx_to_orig_idx[new_idx] = a.GetIdx()
for b in mol0.GetBonds():
    a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    if a1 in orig_idx_to_new_idx and a2 in orig_idx_to_new_idx:
        rw.AddBond(orig_idx_to_new_idx[a1], orig_idx_to_new_idx[a2], b.GetBondType())
heavy_mol = rw.GetMol()

heavy_new_indices = sorted(new_idx_to_orig_idx.keys())
n_heavy = len(heavy_new_indices)
display_labels = {new_idx: str(i) for i, new_idx in enumerate(heavy_new_indices, start=1)}
heavy_orig_indices = [new_idx_to_orig_idx[new_idx] for new_idx in heavy_new_indices]

# force numeric-only labels
for idx, label in display_labels.items():
    heavy_mol.GetAtomWithIdx(idx).SetProp("atomLabel", label)

# ---- load charges and compute std/mean per atom ----
method_std = {}
method_mean = {}
all_means_array = []
for method in methods:
    files = sorted(glob.glob(conf_glob_template.format(method=method)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for method pattern: {conf_glob_template.format(method=method)}")
    arrs = [np.load(f) for f in files]
    arr = np.vstack(arrs)   # shape (n_conf, n_atoms)
    method_std[method] = np.std(arr, axis=0, ddof=0)
    method_mean[method] = np.mean(arr, axis=0)
    all_means_array.append(method_mean[method])

all_means = np.vstack(all_means_array)
means_heavy = all_means[:, heavy_orig_indices]
vmax = np.nanmax(np.abs(means_heavy)) or 1e-6
norm = Normalize(vmin=-vmax, vmax=vmax)
cmap = get_cmap(cmap_name)

# ---- draw molecule with RDKit----
rdDepictor.Compute2DCoords(heavy_mol)
drawer = rdMolDraw2D.MolDraw2DCairo(420, 420)
opts = drawer.drawOptions()
opts.atomLabelFontSize = 18
opts.dotsPerAngstrom = 100
drawer.DrawMolecule(heavy_mol)
drawer.FinishDrawing()
mol_img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")

# ---- determine element colors for legend ----
# get set of atomic numbers present in heavy_mol
elements_present = sorted({a.GetAtomicNum() for a in heavy_mol.GetAtoms()})

# Try several ways to access RDKit palette (different RDKit versions expose it differently).
palette = None
# 1) GetDefaultPalette() function
palfun = getattr(rdMolDraw2D, "GetDefaultPalette", None)
if palfun is not None:
    try:
        palette = palfun()
    except Exception:
        palette = None

# 2) several historically used attribute names
if palette is None:
    for attrname in ("_defaultAtomPalette", "_atomPalette"):
        pal = getattr(rdMolDraw2D, attrname, None)
        if pal is not None:
            palette = pal
            break

# 3) MolDrawOptions().atomColourPalette (some versions)
if palette is None:
    try:
        from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
        tmp = MolDrawOptions()
        palette = getattr(tmp, "atomColourPalette", None)
    except Exception:
        palette = None

# Prepare fallback palette (hex strings) for common elements if RDKit palette is not available.
fallback_palette = {
    6: "#000000",   # C - black
    7: "#3050F8",   # N - blue
    8: "#FF0D0D",   # O - red
    16: "#FFFF30",  # S - yellow
    15: "#FF8000",  # P - orange
    9: "#90E050",   # F - green-ish
    17: "#1FF01F",  # Cl - green
    35: "#A52A2A",  # Br - brown
    53: "#940094",  # I - violet
    1: "#FFFFFF",   # H - white (not used in heavy mol)
}

# Build map atomic_number -> matplotlib-compatible color (RGB tuple or hex)
element_color_map = {}
if palette is not None:
    # palette may be dict-like mapping atomic number -> (r,g,b) either in 0-1 or 0-255
    for Z in elements_present:
        if Z in palette:
            col = palette[Z]
            # col might be tuple of floats (0-1) or ints (0-255) -> convert to hex
            try:
                if all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in col):
                    # convert float rgb to hex
                    rgb = tuple(int(round(255 * float(v))) for v in col)
                else:
                    rgb = tuple(int(v) for v in col)
                hexcol = "#%02x%02x%02x" % rgb
                element_color_map[Z] = hexcol
            except Exception:
                # fallback to fallback_palette if something unexpected
                element_color_map[Z] = fallback_palette.get(Z, None)
        else:
            element_color_map[Z] = fallback_palette.get(Z, None)
else:
    # use fallback palette for present elements; if element not in fallback, use matplotlib tab20
    import matplotlib as mpl
    tab20 = mpl.cm.get_cmap("tab20")
    extra_iter = iter([mpl.colors.to_hex(tab20(i)) for i in range(tab20.N)])
    for Z in elements_present:
        if Z in fallback_palette:
            element_color_map[Z] = fallback_palette[Z]
        else:
            # assign next color from tab20
            try:
                element_color_map[Z] = next(extra_iter)
            except StopIteration:
                # if exhausted, wrap around
                element_color_map[Z] = mpl.colors.to_hex(tab20(Z % tab20.N))

# If any element still has None, assign a neutral gray
for Z in elements_present:
    if element_color_map.get(Z) is None:
        element_color_map[Z] = "#808080"

# ---- build legend handles in a stable order ----
preferred_order = [6, 7, 8, 16, 15]  # C, N, O, S, P
ordered_elements = []
for z in preferred_order:
    if z in elements_present:
        ordered_elements.append(z)
# add remaining elements present sorted ascending
ordered_elements += [z for z in elements_present if z not in ordered_elements]
# build patches
handles = []
ptable = Chem.GetPeriodicTable()
for z in ordered_elements:
    sym = ptable.GetElementSymbol(z)
    color = element_color_map[z]
    handles.append(mpatches.Patch(color=color, label=sym))

# ---- create figure: left mol + legend, right bar chart ----
fig, (ax_mol, ax_chart) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})
ax_mol.imshow(mol_img)
ax_mol.axis('off')
ax_mol.set_title(f"{molecule}",fontsize=16)

# place legend under the molecule image
ax_mol.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=max(1, min(4, len(handles))), fontsize=10, frameon=False) #fontsize was 8

# ---- bar chart: grouped bars (std) colored by mean charge ----
x = np.arange(n_heavy)
bar_width = 0.8 / len(methods)
offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, len(methods))
method_hatches = ['', '//', '..'] * 5
proxies = []
for i, method in enumerate(methods):
    std_heavy = method_std[method][heavy_orig_indices]
    mean_heavy = method_mean[method][heavy_orig_indices]
    xpos = x + offsets[i]
    bar_colors = [cmap(norm(v)) for v in mean_heavy]
    ax_chart.bar(xpos, std_heavy, width=bar_width, color=bar_colors,
                 edgecolor='black', linewidth=0.4, hatch=method_hatches[i])
    proxies.append(mpatches.Rectangle((0, 0), 1, 1, facecolor='white',
                                      edgecolor='black', hatch=method_hatches[i]))

ax_chart.set_xticks(x)
ax_chart.tick_params(axis='y', labelsize=14)
ax_chart.set_xticklabels([str(i) for i in range(1, n_heavy + 1)],fontsize=14)
ax_chart.set_xlabel("Heavy atom (index)",fontsize=16)
ax_chart.set_ylabel("Charge variability (std) [e]",fontsize=16)
ax_chart.set_title("Per-atom variability",fontsize=16)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax_chart, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10) 
cbar.set_label("Mean charge (e)",fontsize = 14)

#ax_chart.legend(proxies, methods, title="Methods")
ax_chart.legend(
    proxies,
    methods,
    title="Methods",
    fontsize=14,          
    title_fontsize=16,    
    loc="upper right",    
    frameon=True           
)
plt.ylim((0,0.20))
plt.tight_layout()
plt.savefig(out_filename, dpi=dpi)
print("Saved", out_filename)