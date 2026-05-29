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

with st.sidebar:
    st.header('Query')
    chrom = st.selectbox('Chromosome (hg38)', CHROMS, index=CHROMS.index('chr4'))
    pos = st.number_input(
        'Position (hg38, 1-based bp)', min_value=1, value=73436824, step=1, format='%d',
    )
    label = st.text_input('Locus label', value='AFP_TSS', max_chars=40)
    cell_type = st.selectbox(
        'Cell type', core.CELL_TYPES, index=core.CELL_TYPES.index('Hepatocytes'),
    )
    window_kb = st.slider('Window (± kb)', 25, 500, 100, step=25)

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


@st.cache_data(max_entries=20, show_spinner=False, ttl=24 * 3600)
def cached_signals(chrom: str, pos: int, cell_type: str, window_kb: float):
    """In-memory cache keyed by (chrom, pos, cell_type, window_kb). Returns dict[str, np.ndarray]."""
    species = core.list_species()
    prog = st.progress(0.0, text=f'Fetching 0/{len(species)} per-species hg38 tracks…')

    def cb(done, total):
        prog.progress(done / total, text=f'Fetching {done}/{total} per-species hg38 tracks…')

    sig = core.fetch_signals(
        chrom, pos, cell_type,
        window_kb=window_kb, bin_kb=0.1,
        species_list=species, max_workers=32,
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
with st.status('Loading…', expanded=False) as status:
    status.update(label='Pulling per-species hg38 tracks…')
    signals = cached_signals(chrom, int(pos), cell_type, float(window_kb))
    status.update(label='Computing synteny + ordering by phylogeny…')
    syn = core.compute_synteny(signals, bin_bp=100.0, max_gap_kb=10.0)
    if show_all:
        retained = syn[syn['coverage_frac'] > 0]['species'].tolist()
    else:
        retained = syn[syn['syntenic_span_kb'] >= min_syn]['species'].tolist()
    order, tree = core.load_pruned_tree(retained)
    highlight = [s.strip() for s in highlight_str.split(',') if s.strip()]
    status.update(label='Rendering figure…')
    fig = core.plot_fig6e(
        order, signals, tree,
        anchor_label=label or f'{chrom}_{pos}',
        anchor_chrom=chrom, anchor_pos=int(pos),
        cell_type=cell_type, window_kb=float(window_kb),
        show_all_species=show_all, min_syntenic_kb=float(min_syn),
        highlight_species=highlight,
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
fname_stem = f'fig6e_{label or chrom}_{cell_type}_w{int(window_kb)}kb'
col1.download_button('Download PNG', png_buf, file_name=fname_stem + '.png',
                     mime='image/png', use_container_width=True)
col2.download_button('Download PDF', pdf_buf, file_name=fname_stem + '.pdf',
                     mime='application/pdf', use_container_width=True)

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
