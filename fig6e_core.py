"""Core logic for the STEAM-v1 Fig 6e per-coordinate viewer.

Pure Python module — no Streamlit dependency. Used by `app.py` and (optionally)
the notebook. All functions take their parameters explicitly; no module globals
encode the current query.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['pdf.fonttype'] = 42

try:
    import pyBigWig
except ImportError as e:
    raise RuntimeError('pyBigWig required: pip install pyBigWig') from e

try:
    from Bio import Phylo
    HAVE_BIO = True
except ImportError:
    HAVE_BIO = False


# --- constants ---------------------------------------------------------------

BASE = 'https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup'
HG38_BW_FMT = BASE + '/jax_atac_augmented_241_mammals_hg38/hg38/{species}/{species}.{cell_type}.bw'
HG38_BW_DIR = BASE + '/jax_atac_augmented_241_mammals_hg38/hg38/'

CELL_TYPES = [
    'Adipocyte_cells', 'Adipocyte_cells_Cyp2e1', 'B_cells',
    'Brain_capillary_endothelial_cells', 'CNS_neurons', 'Cardiomyocytes',
    'Corticofugal_neurons', 'Endocardial_cells', 'Endothelium',
    'Epithelial_cells', 'Erythroid_cells', 'Eye', 'Glia',
    'Glomerular_endothelial_cells', 'Gut_epithelial_cells', 'Hepatocytes',
    'Intermediate_neuronal_progenitors', 'Kidney',
    'Lateral_plate_and_intermediate_mesoderm',
    'Liver_sinusoidal_endothelial_cells', 'Lung_and_airway',
    'Lymphatic_vessel_endothelial_cells', 'Melanocyte_cells', 'Mesoderm',
    'Neural_crest_PNS_neurons', 'Neuroectoderm_and_glia',
    'Olfactory_ensheathing_cells', 'Olfactory_neurons', 'Oligodendrocytes',
    'Skeletal_muscle_cells', 'T_cells', 'White_blood_cells',
]

VIRIDIS_FLOOR = '#440154'

REPO_ROOT = Path(__file__).resolve().parent
SPECIES_INDEX_CACHE = REPO_ROOT / 'cache' / 'species_list.txt'
TREE_PATH_DEFAULT = REPO_ROOT / 'data' / 'zoonomia_241.nwk'


# --- gene symbol -> hg38 TSS (Ensembl REST) ----------------------------------

ENSEMBL_LOOKUP = 'https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}'


def lookup_gene_tss(symbol: str, timeout: float = 15.0) -> dict:
    """Resolve a human gene symbol to its hg38 TSS via Ensembl REST.

    Returns a dict with keys: chrom (str, with 'chr' prefix), tss (int, 1-based),
    strand ('+' or '-'), gene_name, ensembl_id, biotype, start, end.
    Raises ValueError if the symbol is unknown.
    """
    symbol = symbol.strip()
    if not symbol:
        raise ValueError('Empty gene symbol.')
    r = requests.get(
        ENSEMBL_LOOKUP.format(symbol=symbol),
        headers={'Accept': 'application/json'},
        timeout=timeout,
    )
    if r.status_code in (400, 404):
        raise ValueError(f'Gene symbol "{symbol}" not found in Ensembl (human).')
    r.raise_for_status()
    d = r.json()
    seq = d['seq_region_name']
    chrom = seq if seq.startswith('chr') else f'chr{seq}'
    strand = d['strand']  # 1 or -1
    tss = int(d['start']) if strand == 1 else int(d['end'])
    return {
        'chrom': chrom,
        'tss': tss,
        'strand': '+' if strand == 1 else '-',
        'gene_name': d.get('display_name', symbol),
        'ensembl_id': d.get('id'),
        'biotype': d.get('biotype'),
        'start': int(d['start']),
        'end': int(d['end']),
    }


# --- species list ------------------------------------------------------------

def list_species(refresh: bool = False) -> list[str]:
    SPECIES_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SPECIES_INDEX_CACHE.exists() and not refresh:
        return SPECIES_INDEX_CACHE.read_text().split()
    html = requests.get(HG38_BW_DIR, timeout=60).text
    sp = sorted({m for m in re.findall(r'href="([A-Z][A-Za-z_]+)/"', html)})
    SPECIES_INDEX_CACHE.write_text('\n'.join(sp))
    return sp


# --- per-species hg38-projected signal fetch ---------------------------------

def _fetch_one(species: str, chrom: str, start: int, end: int,
               cell_type: str, n_bins: int) -> Optional[np.ndarray]:
    url = HG38_BW_FMT.format(species=species, cell_type=cell_type)
    try:
        bw = pyBigWig.open(url)
    except Exception:
        return None
    try:
        chroms = bw.chroms()
        if chrom not in chroms:
            return None
        end_eff = min(end, chroms[chrom])
        if end_eff <= start:
            return None
        vals = bw.stats(chrom, start, end_eff, type='mean', nBins=n_bins)
    except Exception:
        return None
    finally:
        bw.close()
    return np.array([np.nan if v is None else v for v in vals], dtype=float)


def fetch_signals_multi_ct(chrom: str, pos: int, cell_types: list[str],
                           window_kb: float = 100.0, bin_kb: float = 0.1,
                           species_list: Optional[list[str]] = None,
                           max_workers: int = 64,
                           progress: Optional[Callable[[int, int], None]] = None,
                           ) -> dict[tuple, np.ndarray]:
    """Fetch hg38-projected signals for many (species, cell_type) pairs.

    Returns dict keyed by (species, cell_type) -> 1D array of length n_bins.
    """
    if species_list is None:
        species_list = list_species()
    window_bp = int(window_kb * 1000)
    bin_bp = bin_kb * 1000
    n_bins = int(round(2 * window_bp / bin_bp))
    start = max(0, int(pos) - window_bp)
    end = int(pos) + window_bp

    out: dict[tuple, np.ndarray] = {}
    tasks = [(sp, ct) for sp in species_list for ct in cell_types]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, sp, chrom, start, end, ct, n_bins): (sp, ct)
                for sp, ct in tasks}
        done = 0
        for fut in as_completed(futs):
            sp, ct = futs[fut]
            arr = fut.result()
            if arr is not None and arr.shape == (n_bins,):
                out[(sp, ct)] = arr
            done += 1
            if progress is not None:
                progress(done, len(futs))
    return out


def aggregate_celltype_matrix(signals: dict[tuple, np.ndarray],
                              species_list: list[str],
                              cell_types: list[str],
                              normalisation: str = 'per_row',
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Per (cell_type, bin): sum the raw STEAM-v1 prediction scores across species
    (NaN → 0), then normalise.

    ``normalisation``:
      - 'per_row'  (default): each cell-type row is divided by its own max → reveals
        cell-type-specific patterns; rows are not comparable in magnitude.
      - 'global':  whole matrix divided by global max → preserves cross-cell-type
        magnitudes, but strong broad peaks can dim narrower cell-type-specific peaks.

    Returns (normalised matrix shape (n_ct, n_bins), per-bin synteny coverage shape (n_bins,)).
    """
    n_bins = next(iter(signals.values())).shape[0]
    n_ct = len(cell_types)
    sums = np.zeros((n_ct, n_bins))
    ct_index = {c: i for i, c in enumerate(cell_types)}
    for (sp, ct), arr in signals.items():
        ci = ct_index.get(ct)
        if ci is None:
            continue
        # Raw STEAM-v1 prediction; NaN bins contribute zero (no synteny → no contribution).
        sums[ci] += np.where(np.isfinite(arr), arr, 0.0)

    if normalisation == 'per_row':
        row_max = sums.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        norm_mat = sums / row_max
    elif normalisation == 'row_then_col':
        row_max = sums.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        m1 = sums / row_max
        col_max = m1.max(axis=0, keepdims=True)
        col_max = np.where(col_max > 0, col_max, 1.0)
        norm_mat = m1 / col_max
    elif normalisation == 'col_then_row':
        col_max = sums.max(axis=0, keepdims=True)
        col_max = np.where(col_max > 0, col_max, 1.0)
        m1 = sums / col_max
        row_max = m1.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        norm_mat = m1 / row_max
    else:  # 'global'
        mx = sums.max()
        norm_mat = sums / mx if mx > 0 else sums

    # Synteny coverage per bin: fraction of species with hg38 data, averaged over cell types.
    per_sp_cov = {}
    for (sp, ct), arr in signals.items():
        per_sp_cov.setdefault(sp, []).append(np.isfinite(arr))
    if per_sp_cov:
        species_cov = np.stack([np.mean(np.stack(v), axis=0) for v in per_sp_cov.values()])
        coverage = species_cov.mean(axis=0)
    else:
        coverage = np.zeros(n_bins)
    return norm_mat, coverage


def fetch_signals(chrom: str, pos: int, cell_type: str,
                  window_kb: float = 100.0, bin_kb: float = 0.1,
                  species_list: Optional[list[str]] = None,
                  max_workers: int = 32,
                  progress: Optional[Callable[[int, int], None]] = None,
                  ) -> dict[str, np.ndarray]:
    """Fetch hg38-projected `mean` signal in `bin_kb`-sized bins for each species.

    Returns dict {species: 1D array of length n_bins}, dropping species that
    have no data at the locus.
    """
    if species_list is None:
        species_list = list_species()
    window_bp = int(window_kb * 1000)
    bin_bp = bin_kb * 1000
    n_bins = int(round(2 * window_bp / bin_bp))
    start = max(0, int(pos) - window_bp)
    end = int(pos) + window_bp

    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, sp, chrom, start, end, cell_type, n_bins): sp
                for sp in species_list}
        done = 0
        for fut in as_completed(futs):
            sp = futs[fut]
            arr = fut.result()
            if arr is not None and arr.shape == (n_bins,):
                out[sp] = arr
            done += 1
            if progress is not None:
                progress(done, len(futs))
    return out


# --- synteny proxy -----------------------------------------------------------

def contiguous_span_bins(arr: np.ndarray, max_gap_bins: int) -> int:
    fin = np.isfinite(arr)
    best = cur = gap = 0
    for v in fin:
        if v:
            cur += 1
            gap = 0
        else:
            gap += 1
            cur = cur + 1 if gap <= max_gap_bins else 0
        best = max(best, cur)
    return best


def compute_synteny(signals: dict[str, np.ndarray], bin_bp: float,
                    max_gap_kb: float = 10.0) -> pd.DataFrame:
    max_gap_bins = int(round(max_gap_kb * 1000 / bin_bp))
    rows = []
    for sp, arr in signals.items():
        rows.append({
            'species': sp,
            'syntenic_span_kb': contiguous_span_bins(arr, max_gap_bins) * bin_bp / 1000,
            'coverage_frac': float(np.isfinite(arr).mean()),
            'mean_signal': float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values('syntenic_span_kb', ascending=False)


# --- tree --------------------------------------------------------------------

def ladderize(clade, reverse: bool = True) -> None:
    for c in clade.clades:
        ladderize(c, reverse)
    clade.clades.sort(key=lambda c: c.count_terminals(), reverse=reverse)


def tree_leaf_order(tree_path: Path = TREE_PATH_DEFAULT,
                    restrict_to: Optional[set[str]] = None) -> list[str]:
    """Ladderized leaf order of the Zoonomia tree, optionally restricted to a species set."""
    if not (HAVE_BIO and tree_path.exists()):
        return [] if restrict_to is None else sorted(restrict_to)
    tree = Phylo.read(str(tree_path), 'newick')
    ladderize(tree.root, reverse=True)
    leaves = [t.name for t in tree.get_terminals()]
    if restrict_to is not None:
        leaves = [n for n in leaves if n in restrict_to]
    return leaves


def subsample_evenly(species_pool: list[str], n: int,
                     must_include: tuple[str, ...] = ()) -> list[str]:
    """Pick ~n species evenly along the given order; always keep any must_include present."""
    if n >= len(species_pool):
        return list(species_pool)
    step = len(species_pool) / n
    picked = [species_pool[min(int(i * step), len(species_pool) - 1)] for i in range(n)]
    picked = list(dict.fromkeys(picked))  # de-dup, preserve order
    must = [s for s in must_include if s in species_pool and s not in picked]
    # Insert must-includes at their tree-order position
    pool_idx = {s: i for i, s in enumerate(species_pool)}
    picked = sorted(set(picked) | set(must), key=lambda s: pool_idx[s])
    return picked


def load_pruned_tree(retained_species: list[str],
                     tree_path: Path = TREE_PATH_DEFAULT):
    if not (HAVE_BIO and tree_path.exists()):
        return list(retained_species), None
    tree = Phylo.read(str(tree_path), 'newick')
    keep = set(retained_species)
    for t in [t for t in tree.get_terminals() if t.name not in keep]:
        tree.prune(t)
    ladderize(tree.root, reverse=True)
    order = [t.name for t in tree.get_terminals()]
    order += [s for s in retained_species if s not in order]
    return order, tree


def abbreviate_species(name: str) -> str:
    parts = name.split('_')
    return f'{parts[0][0]}. ' + ' '.join(parts[1:]) if len(parts) >= 2 else name


def _tree_node_coords(tree):
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


def _draw_rect_tree(ax_tree, ax_lab, tree, species_order, fontsize, highlight=()):
    n = len(species_order)
    xs, ys = _tree_node_coords(tree)
    xmax = max(xs.values())

    def seg(clade):
        x0 = xs[id(clade)] - (clade.branch_length or 0.0)
        ax_tree.plot([x0, xs[id(clade)]], [ys[id(clade)], ys[id(clade)]],
                     color='black', lw=0.4, solid_capstyle='butt')
        if not clade.is_terminal():
            cy = [ys[id(c)] for c in clade.clades]
            ax_tree.plot([xs[id(clade)], xs[id(clade)]], [min(cy), max(cy)],
                         color='black', lw=0.4, solid_capstyle='butt')
            for c in clade.clades:
                seg(c)

    seg(tree.root)
    hl = set(highlight)
    for clade in tree.get_terminals():
        y = ys[id(clade)]
        ax_tree.plot([xs[id(clade)], xmax], [y, y], color='0.88', lw=0.2, zorder=0)
        ax_lab.text(0.98, y, abbreviate_species(clade.name), va='center', ha='right',
                    fontsize=fontsize, style='italic',
                    color=('#d62728' if clade.name in hl else 'black'))
    for a in (ax_tree, ax_lab):
        a.set_ylim(n - 0.5, -0.5)
        a.axis('off')
    ax_tree.set_xlim(0, xmax)
    ax_lab.set_xlim(0, 1)


# --- plot --------------------------------------------------------------------

def plot_fig6e(species_order: list[str],
               signals: dict[str, np.ndarray],
               tree_obj,
               *,
               anchor_label: str,
               anchor_chrom: str,
               anchor_pos: int,
               cell_type: str,
               window_kb: float,
               show_all_species: bool,
               min_syntenic_kb: float,
               score_vmax: float = 30.0,
               highlight_species=(),
               calls: Optional[pd.DataFrame] = None,
               anchor_strand: Optional[str] = None):
    n = len(species_order)
    if n == 0:
        fig = plt.figure(figsize=(6, 2))
        fig.text(0.5, 0.5, 'No species to plot.', ha='center', va='center')
        return fig

    have_tree = tree_obj is not None
    have_dots = (calls is not None) and (not calls.empty)
    fontsize = float(np.clip(560 / max(n, 1), 2.0, 7.0))

    mat = np.vstack([signals[s] for s in species_order])
    coverage = np.isfinite(mat).mean(axis=0)
    strength = np.nansum(mat, axis=0)
    strength = strength / strength.max() if strength.max() > 0 else strength
    xgrid = np.linspace(-window_kb, window_kb, mat.shape[1])

    base_h = max(6.0, n * 0.085)
    track_h = 0.75
    # Narrow label column (just wider than longest tip label); near-zero wspace.
    col_w = ([1.0, 0.45] if have_tree else []) + [3.0]
    fig_w = sum(col_w) * 1.7
    fig = plt.figure(figsize=(fig_w, base_h + 2 * track_h + 1.0))
    gs = fig.add_gridspec(3, len(col_w), width_ratios=col_w,
                          height_ratios=[track_h, track_h, base_h],
                          wspace=0.01, hspace=0.05)
    heat_col = len(col_w) - 1

    def style_track(axx, ylabel):
        axx.set_xlim(-window_kb, window_kb)
        axx.set_ylim(0, 1)
        axx.set_xticks([])
        axx.set_yticks([0, 1])
        axx.tick_params(labelsize=6, length=2)
        axx.set_ylabel(ylabel, fontsize=6, rotation=0, ha='right', va='center')
        axx.spines[['top', 'right']].set_visible(False)

    ax_cov = fig.add_subplot(gs[0, heat_col])
    ax_cov.fill_between(xgrid, coverage, color='0.6', lw=0)
    ax_cov.plot(xgrid, coverage, color='black', lw=0.7)
    style_track(ax_cov, 'synteny\n(frac.\nspecies)')
    syn_note = 'all species' if show_all_species else f'syntenic ≥{min_syntenic_kb:g} kb'
    ax_cov.set_title(
        f'{anchor_label}  {anchor_chrom}:{anchor_pos:,}   '
        f'{cell_type}, ±{window_kb:g} kb, {n} species ({syn_note})',
        fontsize=9, pad=16,
    )

    ax_str = fig.add_subplot(gs[1, heat_col], sharex=ax_cov)
    ax_str.fill_between(xgrid, strength, color='#2a788e', lw=0)
    ax_str.plot(xgrid, strength, color='#15616d', lw=0.7)
    style_track(ax_str, 'summed\nstrength\n(norm.)')

    if have_tree:
        ax_tree = fig.add_subplot(gs[2, 0])
        ax_lab = fig.add_subplot(gs[2, 1])
        _draw_rect_tree(ax_tree, ax_lab, tree_obj, species_order, fontsize,
                        highlight_species)

    ax = fig.add_subplot(gs[2, heat_col], sharex=ax_cov)
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(VIRIDIS_FLOOR)
    vmax = float(score_vmax)
    ax.set_facecolor(VIRIDIS_FLOOR)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax,
                   extent=[-window_kb, window_kb, n - 0.5, -0.5],
                   interpolation='nearest')
    if have_dots:
        yidx = {s: i for i, s in enumerate(species_order)}
        d = calls[calls['species'].isin(yidx)]
        ax.scatter(d['rel_pos'] / 1000.0, d['species'].map(yidx),
                   s=10, facecolor='none', edgecolor='red', linewidths=0.5)

    # TSS / anchor marker
    ax.axvline(0, color='white', lw=0.7, ls=':', alpha=0.8)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    arrow_len_kb = max(8.0, window_kb * 0.12)
    if anchor_strand in ('+', '-'):
        dx = arrow_len_kb if anchor_strand == '+' else -arrow_len_kb
        ax.annotate('', xy=(dx, 1.015), xytext=(0, 1.015), xycoords=trans,
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    lw=0.9, mutation_scale=8), annotation_clip=False)
    ax.scatter([0], [1.015], transform=trans, marker='|', s=30, color='black',
               linewidths=1.2, clip_on=False, zorder=5)
    ax.set_yticks([])
    ax.set_xlabel(f'distance from {anchor_label} (kb)', fontsize=9)
    ax.tick_params(labelsize=8)

    if have_tree:
        cb_holder = fig.add_subplot(gs[0:2, 0])
        cb_holder.axis('off')
        cax = cb_holder.inset_axes([0.12, 0.45, 0.85, 0.10])
    else:
        cax = ax.inset_axes([0.0, 1.04, 0.32, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('STEAM-v1 prediction score', fontsize=7, labelpad=2)
    cb.set_ticks([t for t in (0, 10, 20, 30, 40, 50) if t <= vmax + 1e-9])
    cb.ax.tick_params(labelsize=6, length=2)
    return fig


def plot_celltype_view(cell_types: list[str], mat: np.ndarray, coverage: np.ndarray,
                       *,
                       anchor_label: str, anchor_chrom: str, anchor_pos: int,
                       window_kb: float, n_species_used: int,
                       anchor_strand: Optional[str] = None,
                       normalisation_label: str = 'per cell-type row'):
    """Cell-type cross-section view: 32 rows (cell types) × bins heatmap,
    averaged across species. Synteny coverage track on top.
    """
    n_ct = len(cell_types)
    n_bins = mat.shape[1]
    xgrid = np.linspace(-window_kb, window_kb, n_bins)

    base_h = max(5.0, n_ct * 0.18)
    track_h = 0.75
    col_w = [3.4]
    fig_w = 9.2
    fig = plt.figure(figsize=(fig_w, base_h + track_h + 1.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[track_h, base_h], hspace=0.06)

    # --- synteny coverage track ---
    ax_cov = fig.add_subplot(gs[0])
    ax_cov.fill_between(xgrid, coverage, color='0.6', lw=0)
    ax_cov.plot(xgrid, coverage, color='black', lw=0.7)
    ax_cov.set_xlim(-window_kb, window_kb)
    ax_cov.set_ylim(0, 1)
    ax_cov.set_xticks([])
    ax_cov.set_yticks([0, 1])
    ax_cov.tick_params(labelsize=6, length=2)
    ax_cov.set_ylabel('synteny\n(frac.\nspecies)', fontsize=6,
                      rotation=0, ha='right', va='center')
    ax_cov.spines[['top', 'right']].set_visible(False)
    ax_cov.set_title(
        f'{anchor_label}  {anchor_chrom}:{anchor_pos:,}   '
        f'cell-type cross-section, ±{window_kb:g} kb '
        f'(sum of STEAM-v1 prediction scores across {n_species_used} species, '
        f'normalised {normalisation_label})',
        fontsize=9, pad=12,
    )

    # --- 32 × N_BINS heatmap (per-row-normalised sums) ---
    ax = fig.add_subplot(gs[1], sharex=ax_cov)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(VIRIDIS_FLOOR)
    ax.set_facecolor(VIRIDIS_FLOOR)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1.0,
                   extent=[-window_kb, window_kb, n_ct - 0.5, -0.5],
                   interpolation='nearest')

    # TSS line + strand arrow
    ax.axvline(0, color='white', lw=0.7, ls=':', alpha=0.8)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    arrow_len_kb = max(8.0, window_kb * 0.12)
    if anchor_strand in ('+', '-'):
        dx = arrow_len_kb if anchor_strand == '+' else -arrow_len_kb
        ax.annotate('', xy=(dx, 1.015), xytext=(0, 1.015), xycoords=trans,
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    lw=0.9, mutation_scale=8),
                    annotation_clip=False)
    ax.scatter([0], [1.015], transform=trans, marker='|', s=30,
               color='black', linewidths=1.2, clip_on=False, zorder=5)

    ax.set_yticks(range(n_ct))
    ax.set_yticklabels([c.replace('_', ' ') for c in cell_types], fontsize=7)
    # Re-enable x ticks (sharex with ax_cov suppressed them) and label them.
    xticks = np.linspace(-window_kb, window_kb, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(t)}' for t in xticks], fontsize=8)
    ax.set_xlabel(f'distance from {anchor_label} (kb)', fontsize=9)
    ax.tick_params(axis='y', length=0)

    # Compact horizontal colorbar inset at top-right of the heatmap.
    cax = ax.inset_axes([0.85, 1.02, 0.14, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('summed prediction score (normalized)', fontsize=7, labelpad=2)
    cb.set_ticks([0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=6, length=2)
    return fig
