"""Generate fig6e_interactive.ipynb from in-script cell sources.

Running this script rewrites fig6e_interactive.ipynb. Edit the CELLS list below
to change the notebook, then run `python3 build_notebook.py`.
"""
from __future__ import annotations
import os
import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


CELLS = []

CELLS.append(md("""\
# Fig 6e for arbitrary coordinates — STEAM-v1 across 241 Zoonomia mammals

Shendure et al., *evolutionary transfer learning*. Given a single **hg38 anchor**
(e.g. a TSS) and a **cell type**, this notebook reproduces the **Fig 6e**-style view:

- **Left:** Zoonomia phylogenetic tree, pruned to species that retain syntenic continuity
  at the anchor (≥ a kb threshold).
- **Middle:** per-species predicted-accessibility heatmap across the window, on a shared
  **hg38 coordinate axis**.
- **Right (optional):** per-species enhancer-call dots overlaid on the same axis. Requires
  the calls to be in hg38 coordinates (your master synteny graph), supplied via a pluggable
  table. Skipped cleanly if not yet available.

### Key simplification

The lab published each species' STEAM-v1 prediction **already projected into hg38
coordinates** (`jax_atac_augmented_241_mammals_hg38`). So the heatmap and the
syntenic-continuity filter need **no per-species liftover at query time** — every species is
already on the hg38 axis. The only thing that still needs liftover is mapping each species'
native *enhancer calls* onto hg38 for the dot overlay, which is exactly what the master
synteny graph will provide.
"""))

CELLS.append(md("## 1. Setup"))

CELLS.append(code("""\
from __future__ import annotations
import gzip
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42  # editable text in vector exports

try:
    import pyBigWig
    HAVE_BIGWIG = True
except ImportError:
    HAVE_BIGWIG = False
    raise RuntimeError('pyBigWig is required: pip install pyBigWig')

try:
    from Bio import Phylo
    HAVE_BIO = True
except ImportError:
    HAVE_BIO = False
    print('biopython not available - tree panel will be skipped (alphabetical order used).')

PROJECT_DIR = Path('/Users/shendure/Dropbox/claude/interactive_fig_6')
CACHE_DIR = PROJECT_DIR / 'cache'
DATA_DIR = PROJECT_DIR / 'data'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE = 'https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup'

# Per-species predicted accessibility, PROJECTED TO hg38 (shared axis, no lift needed):
HG38_BW_FMT  = BASE + '/jax_atac_augmented_241_mammals_hg38/hg38/{species}/{species}.{cell_type}.bw'
HG38_BW_DIR  = BASE + '/jax_atac_augmented_241_mammals_hg38/hg38/'

# Per-species core enhancer calls, in each species' NATIVE assembly coords:
NATIVE_BED_FMT = BASE + '/jax_atac_augmented_241_mammals_core_enhancer/{species}.core_region.bed.gz'
NATIVE_BED_DIR = BASE + '/jax_atac_augmented_241_mammals_core_enhancer/'

# Human (hg38) reference enhancer calls, for the human row / anchor sanity check:
HUMAN_BED_URL = (BASE.replace('/nobackup', '/backup') +
                 '/jax_atac/download/Supplementary_File_4_Evolution_Augmented_Model_Predict_On_Human_Genome.bed.gz')
"""))

CELLS.append(md("## 2. Query parameters"))

CELLS.append(code("""\
# --- Edit these to query a different locus / cell type ---
ANCHOR_CHROM    = 'chr4'         # hg38 chromosome
ANCHOR_POS      = 73436824       # hg38 1-based position (e.g. AFP TSS)
ANCHOR_LABEL    = 'AFP_TSS'      # used for plot title / output filename
CELL_TYPE       = 'Hepatocytes'  # must match a STEAM-v1 cell class (see list in section 3)
WINDOW_KB       = 100            # plot ± this many kb around the anchor
SHOW_ALL_SPECIES = True          # True: show every species with any hg38 signal (no synteny cut).
                                 # False: restrict to MIN_SYNTENIC_KB (Fig-6d-style fork).
MIN_SYNTENIC_KB = 100            # used only when SHOW_ALL_SPECIES is False
MAX_GAP_KB      = 10             # gaps ≤ this are bridged when measuring contiguous span
BIN_KB          = 0.1           # heatmap bin size (0.1 kb = 100 bp, matching the paper)
SCORE_VMAX_PCT  = 99.5           # clip the viridis color scale at this percentile of score

# Tip labels to colour red on the tree (e.g. focal species). Paper highlights a handful.
HIGHLIGHT_SPECIES = ['Homo_sapiens', 'Mus_musculus']

# Pluggable inputs (zoonomia_241.nwk is shipped; the others are optional):
TREE_PATH       = DATA_DIR / 'zoonomia_241.nwk'         # Newick of the 241 species
HG38_CALLS_TSV  = DATA_DIR / 'enhancer_calls_hg38.tsv'  # per-species calls in hg38 coords (optional)
SYNTENY_TSV     = DATA_DIR / 'syntenic_span.tsv'        # external span table (optional override)
"""))

CELLS.append(md("""\
## 3. Discover the species list

Scraped once from the hg38-projected bigwig directory and cached. This is the set of
**239 non-reference Zoonomia mammals** (human/mouse are the references, distributed
separately).
"""))

CELLS.append(code("""\
SPECIES_CACHE = CACHE_DIR / 'species_list.txt'


def list_species(refresh: bool = False) -> list[str]:
    if SPECIES_CACHE.exists() and not refresh:
        return SPECIES_CACHE.read_text().split()
    html = requests.get(HG38_BW_DIR, timeout=60).text
    species = sorted({m for m in re.findall(r'href=\"([A-Z][A-Za-z_]+)/\"', html)})
    SPECIES_CACHE.write_text('\\n'.join(species))
    return species


SPECIES = list_species()
print(f'{len(SPECIES)} species. First 5: {SPECIES[:5]}')

CELL_TYPES = [
    'Adipocyte_cells','Adipocyte_cells_Cyp2e1','B_cells','Brain_capillary_endothelial_cells',
    'CNS_neurons','Cardiomyocytes','Corticofugal_neurons','Endocardial_cells','Endothelium',
    'Epithelial_cells','Erythroid_cells','Eye','Glia','Glomerular_endothelial_cells',
    'Gut_epithelial_cells','Hepatocytes','Intermediate_neuronal_progenitors','Kidney',
    'Lateral_plate_and_intermediate_mesoderm','Liver_sinusoidal_endothelial_cells',
    'Lung_and_airway','Lymphatic_vessel_endothelial_cells','Melanocyte_cells','Mesoderm',
    'Neural_crest_PNS_neurons','Neuroectoderm_and_glia','Olfactory_ensheathing_cells',
    'Olfactory_neurons','Oligodendrocytes','Skeletal_muscle_cells','T_cells','White_blood_cells',
]
assert CELL_TYPE in CELL_TYPES, f'{CELL_TYPE} not in {CELL_TYPES}'
"""))

CELLS.append(md("""\
## 4. Pull per-species accessibility (hg38 axis) + syntenic-continuity filter

For each species we read its hg38-projected bigwig over `[anchor - WINDOW, anchor + WINDOW]`
in `BIN_KB`-sized bins (`mean` signal; 100 bp matches the paper). pyBigWig reads remotely
over HTTP, so we stream by-bin rather than downloading whole files.

By default (`SHOW_ALL_SPECIES = True`) **every species with any hg38 signal in the window is
shown** — no synteny cut. The per-column synteny coverage is still computed and displayed as
the top track. Set `SHOW_ALL_SPECIES = False` to switch to the Fig-6d-style restricted view
that keeps only species with ≥ `MIN_SYNTENIC_KB` of contiguous syntenic coverage.

**Syntenic continuity** (used by the restricted view, and shown in the `syn` table) is
estimated as the longest run of covered bins around the anchor, bridging internal gaps
≤ `MAX_GAP_KB`. This is a **proxy**; if you have a chain/HAL-derived span table, drop it at
`SYNTENY_TSV` (`species, syntenic_span_kb`) to override it.
"""))

CELLS.append(code("""\
WINDOW_BP = WINDOW_KB * 1000
BIN_BP = BIN_KB * 1000
N_BINS = int(round(2 * WINDOW_BP / BIN_BP))   # e.g. 100 bp bins over ±100 kb -> 2000 bins
START = max(0, ANCHOR_POS - WINDOW_BP)
END = ANCHOR_POS + WINDOW_BP


def contiguous_span_bins(arr: np.ndarray, max_gap_bins: int) -> int:
    \"\"\"Longest run of covered bins, bridging internal gaps up to max_gap_bins.\"\"\"
    fin = np.isfinite(arr)
    best = cur = gap = 0
    for v in fin:
        if v:
            cur += 1
            gap = 0
        else:
            gap += 1
            if gap > max_gap_bins:
                cur = 0
            else:
                cur += 1  # bridge the gap, keep the run alive
        best = max(best, cur)
    return best


def fetch_hg38_signal(species: str) -> Optional[np.ndarray]:
    url = HG38_BW_FMT.format(species=species, cell_type=CELL_TYPE)
    try:
        bw = pyBigWig.open(url)
    except Exception:
        return None
    try:
        chroms = bw.chroms()
        if ANCHOR_CHROM not in chroms:
            return None
        end = min(END, chroms[ANCHOR_CHROM])
        if end <= START:
            return None
        vals = bw.stats(ANCHOR_CHROM, START, end, type='mean', nBins=N_BINS)
    except Exception:
        return None
    finally:
        bw.close()
    return np.array([np.nan if v is None else v for v in vals], dtype=float)


def fetch_all_signals(species_list: list[str], max_workers: int = 12) -> dict[str, np.ndarray]:
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_hg38_signal, s): s for s in species_list}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            arr = fut.result()
            if arr is not None and arr.shape == (N_BINS,):
                out[s] = arr
            if i % 50 == 0:
                print(f'  {i}/{len(species_list)} fetched')
    return out


print(f'Fetching hg38-projected {CELL_TYPE} signal for {len(SPECIES)} species '
      f'({ANCHOR_CHROM}:{START:,}-{END:,}, {N_BINS} bins)...')
signals = fetch_all_signals(SPECIES)
print(f'Got signal for {len(signals)} species.')
"""))

CELLS.append(code("""\
# Apply syntenic-continuity filter.
max_gap_bins = int(round(MAX_GAP_KB * 1000 / BIN_BP))
syn_rows = []
for sp, arr in signals.items():
    span_bins = contiguous_span_bins(arr, max_gap_bins)
    syn_rows.append({'species': sp,
                     'syntenic_span_kb': span_bins * BIN_BP / 1000,
                     'coverage_frac': float(np.isfinite(arr).mean()),
                     'mean_signal': float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan})
syn = pd.DataFrame(syn_rows)

# Optional override from an external chain/HAL-derived span table.
if SYNTENY_TSV.exists():
    ext = pd.read_csv(SYNTENY_TSV, sep='\\t')[['species', 'syntenic_span_kb']]
    syn = syn.drop(columns='syntenic_span_kb').merge(ext, on='species', how='left')
    print(f'Using external syntenic spans from {SYNTENY_TSV.name}.')

syn = syn.sort_values('syntenic_span_kb', ascending=False)
if SHOW_ALL_SPECIES:
    # Show every species that has ANY hg38 signal in the window (drop only all-NaN rows).
    retained = syn[syn['coverage_frac'] > 0]['species'].tolist()
    print(f'Showing all {len(retained)} species with hg38 signal '
          f'(synteny filter OFF; SHOW_ALL_SPECIES=True).')
else:
    retained = syn[syn['syntenic_span_kb'] >= MIN_SYNTENIC_KB]['species'].tolist()
    print(f'{len(retained)}/{len(signals)} species pass MIN_SYNTENIC_KB={MIN_SYNTENIC_KB} '
          f'(gap tolerance {MAX_GAP_KB} kb).')
syn.head(10)
"""))

CELLS.append(md("""\
## 5. (Optional) Enhancer-call dots in hg38 coordinates

Fig 6e overlays the actual core-enhancer calls. Those calls live in each species' **native**
assembly, so they need to be lifted onto hg38 to share the heatmap axis — that's the job of
the master synteny graph being built separately.

This cell reads an optional TSV `HG38_CALLS_TSV` with columns:
`species, hg38_start, hg38_end, phred`. If absent, dots are skipped (heatmap still renders).
"""))

CELLS.append(code("""\
def load_hg38_calls(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f'No hg38 calls table at {path} - dot overlay will be skipped.')
        return pd.DataFrame(columns=['species', 'hg38_start', 'hg38_end', 'phred'])
    df = pd.read_csv(path, sep='\\t')
    need = {'species', 'hg38_start', 'hg38_end', 'phred'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {missing}')
    return df


calls = load_hg38_calls(HG38_CALLS_TSV)
if not calls.empty:
    mid = (calls['hg38_start'] + calls['hg38_end']) // 2
    calls = calls.assign(rel_pos=(mid - ANCHOR_POS).astype(int))
    calls = calls[(calls['rel_pos'].abs() <= WINDOW_BP) & (calls['species'].isin(retained))]
print(f'{len(calls)} enhancer-call dots in window.' if not calls.empty else 'No dots.')
"""))

CELLS.append(md("""\
## 6. Phylogenetic tree — row order + tree panel (default ON)

The Zoonomia 241-mammal tree (`data/zoonomia_241.nwk`, extracted from the same
`241-mammalian-2020v2.hal` the projections were built on) is pruned to retained species and
**ladderized**. Its leaf order drives the heatmap rows, and it is drawn as a rectangular
phylogram to the left of the heatmap (matching Fig 6e). If the tree file is missing, rows
fall back to syntenic-span order and the tree panel is skipped.
"""))

CELLS.append(code("""\
def ladderize(clade, reverse=True):
    for c in clade.clades:
        ladderize(c, reverse)
    clade.clades.sort(key=lambda c: c.count_terminals(), reverse=reverse)


def load_pruned_tree(tree_path: Path, retained_species: list[str]):
    if not (HAVE_BIO and tree_path.exists()):
        print(f'No tree at {tree_path} - ordering rows by syntenic span, no tree panel.')
        order = [s for s in syn['species'].tolist() if s in set(retained_species)]
        return order, None
    tree = Phylo.read(str(tree_path), 'newick')
    keep = set(retained_species)
    for t in [t for t in tree.get_terminals() if t.name not in keep]:
        tree.prune(t)
    ladderize(tree.root, reverse=True)
    order = [t.name for t in tree.get_terminals()]
    order += [s for s in retained_species if s not in order]  # any leaves missing from tree
    return order, tree


def abbreviate_species(name: str) -> str:
    \"\"\"Macaca_mulatta -> 'M. mulatta'; Canis_lupus_familiaris -> 'C. lupus familiaris'.\"\"\"
    parts = name.split('_')
    if len(parts) >= 2:
        return f'{parts[0][0]}. ' + ' '.join(parts[1:])
    return name


def tree_node_coords(tree):
    \"\"\"x = root-to-node distance (branch lengths); y = leaf index / mean of children.\"\"\"
    leaves = tree.get_terminals()
    leaf_y = {id(l): i for i, l in enumerate(leaves)}
    xs, ys = {}, {}

    def depth(clade, x0):
        x = x0 + (clade.branch_length or 0.0)
        xs[id(clade)] = x
        if clade.is_terminal():
            ys[id(clade)] = leaf_y[id(clade)]
        else:
            for c in clade.clades:
                depth(c, x)
            ys[id(clade)] = float(np.mean([ys[id(c)] for c in clade.clades]))

    depth(tree.root, 0.0)
    return xs, ys


species_order, tree_obj = load_pruned_tree(TREE_PATH, retained)
print(f'{len(species_order)} rows; tree panel: {tree_obj is not None}')
print('top of tree:', species_order[:6])
"""))

CELLS.append(md("## 7. Plot"))

CELLS.append(code("""\
VIRIDIS_FLOOR = '#440154'  # paper uses this as the panel background / NA colour


def draw_rect_tree(ax_tree, ax_lab, tree, species_order, fontsize, highlight=()):
    \"\"\"Branches in ax_tree; abbreviated tip labels in the dedicated ax_lab column.\"\"\"
    n = len(species_order)
    xs, ys = tree_node_coords(tree)
    xmax = max(xs.values())

    def seg(clade):
        x0 = xs[id(clade)] - (clade.branch_length or 0.0)
        ax_tree.plot([x0, xs[id(clade)]], [ys[id(clade)], ys[id(clade)]],
                     color='black', lw=0.4, solid_capstyle='butt')      # horizontal branch
        if not clade.is_terminal():
            cy = [ys[id(c)] for c in clade.clades]
            ax_tree.plot([xs[id(clade)], xs[id(clade)]], [min(cy), max(cy)],
                         color='black', lw=0.4, solid_capstyle='butt')  # vertical connector
            for c in clade.clades:
                seg(c)

    seg(tree.root)

    hl = set(highlight)
    for clade in tree.get_terminals():
        y = ys[id(clade)]
        ax_tree.plot([xs[id(clade)], xmax], [y, y], color='0.88', lw=0.2, zorder=0)  # guide
        ax_lab.text(0.98, y, abbreviate_species(clade.name), va='center', ha='right',
                    fontsize=fontsize, style='italic',
                    color=('#d62728' if clade.name in hl else 'black'))

    for a in (ax_tree, ax_lab):
        a.set_ylim(n - 0.5, -0.5)
        a.axis('off')
    ax_tree.set_xlim(0, xmax)
    ax_lab.set_xlim(0, 1)


def plot_fig6e(species_order, signals, calls, tree_obj, outpath=None):
    n = len(species_order)
    if n == 0:
        print('No species to plot.')
        return None
    have_tree = tree_obj is not None
    have_dots = (calls is not None) and (not calls.empty)
    fontsize = float(np.clip(560 / max(n, 1), 2.0, 7.0))

    mat = np.vstack([signals[s] for s in species_order])
    coverage = np.isfinite(mat).mean(axis=0)              # syntenic retention per bin
    strength = np.nansum(mat, axis=0)                     # summed predicted accessibility
    strength = strength / strength.max() if strength.max() > 0 else strength  # normalise 0-1
    xgrid = np.linspace(-WINDOW_KB, WINDOW_KB, mat.shape[1])

    base_h = max(6.0, n * 0.085)
    track_h = 0.75
    col_w = ([1.0, 0.9] if have_tree else []) + [3.0]
    fig_w = sum(col_w) * 1.55
    fig = plt.figure(figsize=(fig_w, base_h + 2 * track_h + 1.2))
    # rows: [coverage track, strength track, main]
    gs = fig.add_gridspec(
        3, len(col_w), width_ratios=col_w, height_ratios=[track_h, track_h, base_h],
        wspace=0.02, hspace=0.18,
    )
    heat_col = len(col_w) - 1

    def style_track(axx, ylabel):
        axx.set_xlim(-WINDOW_KB, WINDOW_KB)
        axx.set_ylim(0, 1)
        axx.set_xticks([])
        axx.set_yticks([0, 1])
        axx.tick_params(labelsize=6, length=2)
        axx.set_ylabel(ylabel, fontsize=6, rotation=0, ha='right', va='center')
        axx.spines[['top', 'right']].set_visible(False)

    # --- track 1: synteny coverage ---
    ax_cov = fig.add_subplot(gs[0, heat_col])
    ax_cov.fill_between(xgrid, coverage, color='0.6', lw=0)
    ax_cov.plot(xgrid, coverage, color='black', lw=0.7)
    style_track(ax_cov, 'synteny\\n(frac.\\nspecies)')
    syn_note = 'all species' if SHOW_ALL_SPECIES else f'syntenic \\u2265{MIN_SYNTENIC_KB} kb'
    ax_cov.set_title(f'{ANCHOR_LABEL}  {ANCHOR_CHROM}:{ANCHOR_POS:,}   '
                     f'{CELL_TYPE}, \\u00b1{WINDOW_KB} kb, {n} species ({syn_note})',
                     fontsize=9, pad=16)

    # --- track 2: normalised summed strength across species ---
    ax_str = fig.add_subplot(gs[1, heat_col], sharex=ax_cov)
    ax_str.fill_between(xgrid, strength, color='#2a788e', lw=0)
    ax_str.plot(xgrid, strength, color='#15616d', lw=0.7)
    style_track(ax_str, 'summed\\nstrength\\n(norm.)')

    # --- tree + labels (aligned to the heatmap row only) ---
    if have_tree:
        ax_tree = fig.add_subplot(gs[2, 0])
        ax_lab = fig.add_subplot(gs[2, 1])
        draw_rect_tree(ax_tree, ax_lab, tree_obj, species_order, fontsize,
                       highlight=HIGHLIGHT_SPECIES)

    # --- heatmap ---
    ax = fig.add_subplot(gs[2, heat_col], sharex=ax_cov)
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(VIRIDIS_FLOOR)
    finite = mat[np.isfinite(mat)]
    vmax = np.percentile(finite, SCORE_VMAX_PCT) if finite.size else 1.0
    ax.set_facecolor(VIRIDIS_FLOOR)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax,
                   extent=[-WINDOW_KB, WINDOW_KB, n - 0.5, -0.5], interpolation='nearest')
    if have_dots:
        yidx = {s: i for i, s in enumerate(species_order)}
        d = calls[calls['species'].isin(yidx)]
        ax.scatter(d['rel_pos'] / 1000.0, d['species'].map(yidx),
                   s=10, facecolor='none', edgecolor='red', linewidths=0.5)
    ax.set_yticks([])
    ax.set_xlabel(f'distance from {ANCHOR_LABEL} (kb)', fontsize=9)
    ax.tick_params(labelsize=8)

    # --- small horizontal colorbar, tucked into the empty top-left ---
    if have_tree:
        cb_holder = fig.add_subplot(gs[0:2, 0]); cb_holder.axis('off')
        cax = cb_holder.inset_axes([0.12, 0.45, 0.85, 0.10])
    else:
        cb_holder = ax; cax = ax.inset_axes([0.0, 1.04, 0.32, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('STEAM-v1 score', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2)

    if outpath:
        fig.savefig(outpath, dpi=220, bbox_inches='tight')
        print(f'Saved -> {outpath}')
    return fig


OUT = PROJECT_DIR / f'fig6e_{ANCHOR_LABEL}_{CELL_TYPE}_w{WINDOW_KB}kb.png'
fig = plot_fig6e(species_order, signals, calls, tree_obj, outpath=OUT)
plt.show()
"""))

CELLS.append(md("""\
## What to wire in next

1. **`HG38_CALLS_TSV`** — per-species core-enhancer calls lifted to hg38
   (`species, hg38_start, hg38_end, phred`), from your master synteny graph, to overlay the
   Fig-6e dots. The heatmap already works without it.
2. To recreate **Fig 6d** (synteny network) and **6f** (per-cluster stratification), feed the
   same hg38 calls table into a graph build (overlap → connected components → cluster) — a
   natural follow-on once the synteny graph exists.

### Notes / caveats
- The syntenic-continuity filter here is a **coverage proxy** from the hg38-projected track,
  not a formal alignment-block length. It correlates with synteny but a species with a real
  alignment gap inside the window will be penalized. Swap in a chain/HAL-derived span if you
  want the exact metric.
- Heatmap rows where a species lacks hg38 coverage appear blank (NaN) within the window.
"""))


def build():
    nb = nbf.v4.new_notebook()
    nb['cells'] = CELLS
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig6e_interactive.ipynb')
    with open(out, 'w') as f:
        nbf.write(nb, f)
    print('Wrote', out)


if __name__ == '__main__':
    build()
