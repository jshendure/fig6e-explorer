"""Streamlit app: STEAM-v1 cross-species Fig 6e explorer.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    https://share.streamlit.io  -> point at this repo + branch + app.py.
"""
from __future__ import annotations

import io
import time

import streamlit as st

import fig6e_core as core

st.set_page_config(
    page_title='STEAM-v1 Fig 6e explorer',
    page_icon='🧬',
    layout='wide',
)

st.title('STEAM-v1 cross-species enhancer view')
st.caption(
    'Recreate Fig 6e from Shendure et al., '
    '[*evolutionary transfer learning*](https://shendure.curve.space/articles/evolutionary-transfer-learning), '
    'for arbitrary hg38 coordinates. Each row is one of 239 Zoonomia mammals; '
    'columns are predicted STEAM-v1 accessibility in 100 bp bins around the anchor.'
)

CHROMS = [f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y']]

@st.cache_data(max_entries=200, show_spinner=False, ttl=24 * 3600)
def cached_gene_lookup(symbol: str):
    return core.lookup_gene_tss(symbol)


with st.sidebar:
    st.header('Query')
    view_mode = st.radio(
        'View', ['Per-species (locus)', 'Cell-type cross-section'], horizontal=False,
        help='Per-species: rows are species, one cell type. '
             'Cell-type cross-section: rows are 32 cell types, signal averaged across species.',
    )
    if 'mode' not in st.session_state:
        st.session_state['mode'] = 'Gene symbol'
    mode = st.radio(
        'Look up by:', ['Gene symbol', 'Coordinate'], horizontal=True, key='mode',
        help='Gene symbol → canonical TSS from Ensembl (hg38). '
             'Coordinate → exact hg38 position you provide.',
    )

    if mode == 'Gene symbol':
        if 'gene_sym_val' not in st.session_state:
            st.session_state['gene_sym_val'] = 'GATA4'
        gene_sym = st.text_input(
            'Gene symbol (HGNC, e.g. GATA4, AFP, FOXP3)', max_chars=40,
            key='gene_sym_val',
        ).strip().upper()
        chrom, pos, label = None, None, None  # resolved on Generate
    else:
        chrom = st.selectbox('Chromosome (hg38)', CHROMS, index=CHROMS.index('chr4'))
        pos = st.number_input(
            'Position (hg38, 1-based bp)', min_value=1, value=73436824, step=1, format='%d',
        )
        label = st.text_input('Locus label', value='AFP_TSS', max_chars=40)
        gene_sym = None

    if view_mode == 'Per-species (locus)':
        cell_type = st.selectbox(
            'Cell type', core.CELL_TYPES, index=core.CELL_TYPES.index('Hepatocytes'),
        )
    else:
        cell_type = 'Hepatocytes'  # placeholder; not used in cross-section view
        st.caption('Cell type fixed in cross-section view (all 32 are shown).')
    window_kb = st.slider('Window (± kb)', 25, 500, 100, step=25)
    n_species = st.select_slider(
        'Species to fetch (fewer = faster)',
        options=[30, 60, 120, 'All (239)'], value=30,
        help='Subsampled evenly along the Zoonomia phylogeny; highlighted species are always kept.',
    )
    if view_mode == 'Cell-type cross-section':
        norm_choice = st.radio(
            'Heatmap normalisation',
            ['Per row', 'Per row → per column', 'Per column → per row', 'Global max'],
            index=1,
            help=('Per row: each cell-type row scaled to [0,1] by its own max — shows '
                  'each cell type\'s pattern. Per row → per column: two-stage, '
                  'highlights which cell type dominates each position. Global max: '
                  'preserves cross-cell-type magnitudes.'),
        )
    else:
        norm_choice = 'Per row'

    st.divider()
    show_all = st.toggle(
        'Show all species (no synteny cut)', value=True,
        help='Off → keep only species with ≥ Min syntenic span of contiguous hg38 coverage.',
    )
    min_syn = st.slider(
        'Min syntenic span (kb)', 0, 200, 100, step=10, disabled=show_all,
    )
    highlight_str = st.text_input(
        'Highlight species (red labels, comma-separated)',
        value='Homo_sapiens, Mus_musculus',
        help='Underscored species names. Species not in the 239-mammal set are silently ignored.',
    )

    run = st.button('Generate figure', type='primary', use_container_width=True)

    # --- Top-TFs quick picker ---
    @st.cache_data(show_spinner=False)
    def _load_top10():
        import pandas as pd, pathlib
        p = pathlib.Path('data/tf_top10_per_celltype.tsv')
        return pd.read_csv(p, sep='\t') if p.exists() else None

    _top = _load_top10()
    if _top is not None:
        sub = _top[_top['cell_type'] == cell_type].sort_values('rank').head(10)
        with st.expander(f'💡 Top TFs in {cell_type} (click to render)', expanded=False):
            st.caption(
                'Per-TF specificity = average across 30 phylogenetically-spread '
                'species of (fraction of total accessibility at the TF locus in this '
                'cell type). Click any to switch the gene symbol + render.'
            )
            for _, row in sub.iterrows():
                if st.button(
                    f"{int(row['rank'])}. {row['tf_symbol']}  "
                    f"(frac {row['rank_score']:.3f}, τ {row['mean_tau']:.2f})",
                    key=f"toptf_{cell_type}_{row['tf_symbol']}",
                    use_container_width=True,
                ):
                    st.session_state['gene_sym_val'] = row['tf_symbol']
                    st.session_state['mode'] = 'Gene symbol'
                    st.session_state['auto_run'] = True
                    st.rerun()

# Honour auto_run requested by Top-TFs picker.
if st.session_state.pop('auto_run', False):
    run = True


@st.cache_data(max_entries=10, show_spinner=False, ttl=24 * 3600)
def cached_signals_multi_ct(chrom: str, pos: int, window_kb: float,
                            species_tuple: tuple, cell_types_tuple: tuple):
    """Fetch all (species, cell_type) pairs for the locus."""
    species = list(species_tuple); cts = list(cell_types_tuple)
    n = len(species) * len(cts)
    prog = st.progress(0.0, text=f'Fetching 0/{n} (species × cell-type) tracks…')

    def cb(done, total):
        prog.progress(done / total, text=f'Fetching {done}/{total} tracks…')

    out = core.fetch_signals_multi_ct(
        chrom, pos, cts, window_kb=window_kb, bin_kb=0.1,
        species_list=species, max_workers=64, progress=cb,
    )
    prog.empty()
    return out


@st.cache_data(max_entries=20, show_spinner=False, ttl=24 * 3600)
def cached_signals(chrom: str, pos: int, cell_type: str, window_kb: float,
                   species_tuple: tuple):
    """Cache key: (chrom, pos, cell_type, window_kb, species set)."""
    species = list(species_tuple)
    prog = st.progress(0.0, text=f'Fetching 0/{len(species)} per-species hg38 tracks…')

    def cb(done, total):
        prog.progress(done / total, text=f'Fetching {done}/{total} per-species hg38 tracks…')

    sig = core.fetch_signals(
        chrom, pos, cell_type,
        window_kb=window_kb, bin_kb=0.1,
        species_list=species, max_workers=64,
        progress=cb,
    )
    prog.empty()
    return sig


if not run:
    st.info(
        'Set query parameters in the sidebar and click **Generate figure**. '
        'The first hit on a fresh locus takes ~30–60 s (239 remote bigwig reads); '
        'repeats are instant from cache.'
    )
    st.stop()

t0 = time.time()
anchor_strand = None

if mode == 'Gene symbol':
    if not gene_sym:
        st.error('Enter a gene symbol.')
        st.stop()
    try:
        info = cached_gene_lookup(gene_sym)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    chrom = info['chrom']
    pos = info['tss']
    label = info['gene_name']
    anchor_strand = info['strand']
    st.info(
        f"**{info['gene_name']}** → `{chrom}:{pos:,}` "
        f"({info['strand']} strand, {info['biotype']}, Ensembl `{info['ensembl_id']}`)"
    )

with st.status('Loading…', expanded=False) as status:
    status.update(label='Picking species subset…')
    all_species = set(core.list_species())
    species_in_tree_order = core.tree_leaf_order(restrict_to=all_species)
    highlight = [s.strip() for s in highlight_str.split(',') if s.strip()]
    if n_species == 'All (239)':
        species_to_fetch = species_in_tree_order
    else:
        species_to_fetch = core.subsample_evenly(
            species_in_tree_order, int(n_species),
            must_include=tuple(s for s in highlight if s in all_species),
        )

    if view_mode == 'Cell-type cross-section':
        status.update(label=f'Pulling {len(species_to_fetch)}×{len(core.CELL_TYPES)} '
                              '(species × cell-type) tracks…')
        signals_mc = cached_signals_multi_ct(
            chrom, int(pos), float(window_kb),
            tuple(species_to_fetch), tuple(core.CELL_TYPES),
        )
        status.update(label='Aggregating per cell type…')
        norm_map = {'Per row': 'per_row',
                    'Per row → per column': 'row_then_col',
                    'Per column → per row': 'col_then_row',
                    'Global max': 'global'}
        mat_ct, coverage = core.aggregate_celltype_matrix(
            signals_mc, list(species_to_fetch), list(core.CELL_TYPES),
            normalisation=norm_map[norm_choice],
        )
        status.update(label='Rendering figure…')
        norm_label_map = {
            'per_row': 'per cell-type row',
            'row_then_col': 'per row, then per column',
            'col_then_row': 'per column, then per row',
            'global': 'by global max',
        }
        fig = core.plot_celltype_view(
            list(core.CELL_TYPES), mat_ct, coverage,
            anchor_label=label or f'{chrom}_{pos}',
            anchor_chrom=chrom, anchor_pos=int(pos),
            window_kb=float(window_kb),
            n_species_used=len(species_to_fetch),
            anchor_strand=anchor_strand,
            normalisation_label=norm_label_map[norm_map[norm_choice]],
        )
        syn = None
    else:
        status.update(label=f'Pulling {len(species_to_fetch)} per-species hg38 tracks…')
        signals = cached_signals(chrom, int(pos), cell_type, float(window_kb),
                                 tuple(species_to_fetch))
        status.update(label='Computing synteny + ordering by phylogeny…')
        syn = core.compute_synteny(signals, bin_bp=100.0, max_gap_kb=10.0)
        if show_all:
            retained = syn[syn['coverage_frac'] > 0]['species'].tolist()
        else:
            retained = syn[syn['syntenic_span_kb'] >= min_syn]['species'].tolist()
        order, tree = core.load_pruned_tree(retained)
        status.update(label='Rendering figure…')
        fig = core.plot_fig6e(
            order, signals, tree,
            anchor_label=label or f'{chrom}_{pos}',
            anchor_chrom=chrom, anchor_pos=int(pos),
            cell_type=cell_type, window_kb=float(window_kb),
            show_all_species=show_all, min_syntenic_kb=float(min_syn),
            highlight_species=highlight,
            anchor_strand=anchor_strand,
        )
    status.update(label=f'Done in {time.time() - t0:.1f} s', state='complete')

st.pyplot(fig, use_container_width=True)

col1, col2, _ = st.columns([1, 1, 4])
png_buf = io.BytesIO()
fig.savefig(png_buf, format='png', dpi=200, bbox_inches='tight')
png_buf.seek(0)
pdf_buf = io.BytesIO()
fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
pdf_buf.seek(0)
view_suffix = 'crosssec' if view_mode == 'Cell-type cross-section' else cell_type
fname_stem = f'fig6e_{label or chrom}_{view_suffix}_w{int(window_kb)}kb'
col1.download_button('Download PNG', png_buf, file_name=fname_stem + '.png',
                     mime='image/png', use_container_width=True)
col2.download_button('Download PDF', pdf_buf, file_name=fname_stem + '.pdf',
                     mime='application/pdf', use_container_width=True)

if syn is not None:
    with st.expander(f'Per-species synteny table ({len(syn)} species)'):
        st.dataframe(
            syn.reset_index(drop=True).round({'syntenic_span_kb': 1, 'coverage_frac': 3,
                                              'mean_signal': 3}),
            use_container_width=True, height=400,
        )

st.caption(
    'Data: Shendure lab `jax_atac_augmented_241_mammals_hg38` hg38-projected STEAM-v1 '
    'tracks. Phylogeny: `halStats --tree` on the Zoonomia 241-mammalian-2020v2 Cactus '
    'alignment. The synteny track here is a coverage proxy from the hg38 projection — '
    'not a formal chain/HAL-derived span.'
)
