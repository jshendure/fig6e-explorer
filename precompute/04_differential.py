"""For each cell type, compute its top-K most similar neighbours and the TFs that
distinguish it from those neighbours.

Inputs:
  data/tf_specificity.parquet  (from 03_specificity.py)

Outputs:
  data/celltype_neighbors.tsv               (per cell type, K nearest by Pearson on mean_signal)
  data/tf_top10_differential_per_celltype.tsv  (top 10 TFs by differential score)

Differential score per (TF, cell_type) =
    mean_signal_in_this_ct  -  mean(mean_signal in the K nearest cell types)

Filter: same MIN_SPECIES_WITH_DATA gate as 03_specificity.py.
We also require the TF to be above the median signal in this cell type, so we don't
surface TFs that are merely *absent* in the neighbours.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / 'data' / 'tf_specificity.parquet'
NEIGHBORS_TSV = REPO / 'data' / 'celltype_neighbors.tsv'
OUT_TOP10 = REPO / 'data' / 'tf_top10_differential_per_celltype.tsv'

K_NEIGHBORS = 5
MIN_SPECIES_WITH_DATA = 20


def main():
    if not SPEC.exists():
        sys.exit(f'Missing {SPEC}; run 03_specificity.py first.')
    sp = pd.read_parquet(SPEC)
    sp = sp[sp['n_species_with_data'] >= MIN_SPECIES_WITH_DATA].copy()

    wide = sp.pivot_table(index='tf_symbol', columns='cell_type', values='mean_signal')
    print(f'TF-by-cell-type matrix: {wide.shape}')

    # 1) Per-cell-type neighbours by Pearson correlation across the 1622-TF vector.
    corr = wide.corr()
    nbr_rows = []
    for ct in corr.columns:
        nearest = corr[ct].drop(ct).sort_values(ascending=False).head(K_NEIGHBORS)
        for i, (n, r) in enumerate(nearest.items(), 1):
            nbr_rows.append({'cell_type': ct, 'rank': i, 'neighbor': n, 'pearson_r': float(r)})
    nbrs = pd.DataFrame(nbr_rows)
    nbrs.to_csv(NEIGHBORS_TSV, sep='\t', index=False, float_format='%.4f')
    print(f'Wrote {NEIGHBORS_TSV}')

    # 2) Differential score per (TF, cell_type) vs. the K nearest cell types.
    diff_rows = []
    median_per_ct = wide.median(axis=0)  # per-CT median signal for the median-floor filter
    for ct in corr.columns:
        neighbours = nbrs[nbrs['cell_type'] == ct]['neighbor'].tolist()
        this_sig = wide[ct]
        nbr_mean = wide[neighbours].mean(axis=1)
        diff = this_sig - nbr_mean
        # Require TF signal in this CT to be above CT median (so we surface
        # what's expressed here, not what's missing elsewhere).
        keep = this_sig >= median_per_ct[ct]
        ranked = diff[keep].sort_values(ascending=False).head(10)
        for rank, (tf, d) in enumerate(ranked.items(), 1):
            diff_rows.append({
                'cell_type': ct, 'rank': rank, 'tf_symbol': tf,
                'differential': float(d),
                'mean_signal_here':      float(this_sig[tf]),
                'mean_signal_neighbors': float(nbr_mean[tf]),
                'neighbors': ','.join(neighbours),
            })
    out = pd.DataFrame(diff_rows)
    out.to_csv(OUT_TOP10, sep='\t', index=False, float_format='%.4f')
    print(f'Wrote {OUT_TOP10}  ({len(out)} rows)')

    print('\nSample (top-3 differential per endothelial cell type):')
    endo = [c for c in corr.columns if 'endothel' in c.lower() or c == 'Endocardial_cells']
    sample = out[out['cell_type'].isin(endo) & (out['rank'] <= 3)]
    print(sample[['cell_type', 'rank', 'tf_symbol', 'differential',
                  'mean_signal_here', 'mean_signal_neighbors']].to_string(index=False))


if __name__ == '__main__':
    main()
