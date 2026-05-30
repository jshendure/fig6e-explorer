"""Compute TF cross-species cell-type specificity from data/tf_window_sums.parquet.

Metrics:
  tau          : Yanai 2005 tissue-specificity index per (tf, species), averaged across
                 species. tau in [0, 1]; 1 = perfectly specific to one cell type, 0 = flat.
  argmax_frac  : per (tf, cell_type) — fraction of species where this cell type is the
                 highest-signal cell type. Identifies consistent cross-species dominance.
  mean_frac_in : per (tf, cell_type) — average across species of (signal_ct / sum_signal_ct).
                 0..1, fraction of the TF's total accessibility in this cell type.
  mean_signal  : per (tf, cell_type) — average signal across species. Magnitude.

Outputs:
  data/tf_specificity.parquet         (long form; tf_symbol, cell_type, all metrics)
  data/tf_top10_per_celltype.tsv      (top 10 TFs per cell type by argmax_frac*tau)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SUMS_PARQUET = REPO / 'data' / 'tf_window_sums.parquet'
OUT_PARQUET = REPO / 'data' / 'tf_specificity.parquet'
OUT_TOP10 = REPO / 'data' / 'tf_top10_per_celltype.tsv'


def tau_index(values: np.ndarray) -> float:
    """Yanai tissue specificity index, [0,1]. NaN-safe."""
    v = values[np.isfinite(values)]
    if len(v) < 2 or v.max() <= 0:
        return np.nan
    return float(np.sum(1 - v / v.max()) / (len(v) - 1))


def main():
    if not SUMS_PARQUET.exists():
        sys.exit(f'Missing {SUMS_PARQUET}; run 02_compute_sums.py first.')
    df = pd.read_parquet(SUMS_PARQUET)
    print(f'{len(df):,} rows; {df["tf_symbol"].nunique()} TFs, '
          f'{df["species"].nunique()} species, {df["cell_type"].nunique()} cell types.')

    metric_col = 'mean'  # length-fair; sum is also in the parquet if you prefer

    # Pivot to (tf, species) x cell_type
    wide = df.pivot_table(index=['tf_symbol', 'species'], columns='cell_type',
                          values=metric_col, aggfunc='first')
    cts = list(wide.columns)
    n_ct = len(cts)
    print(f'Wide matrix: {wide.shape} (tf*species, cell_types). Computing metrics…')

    # Per (tf, species) Tau and argmax cell type
    arr = wide.to_numpy(dtype=float)
    tau_per_row = np.array([tau_index(r) for r in arr])
    argmax_per_row = np.array([cts[i] if np.isfinite(arr[k]).any() else None
                               for k, i in enumerate(np.nanargmax(np.where(np.isnan(arr), -np.inf, arr), axis=1))])
    # Per (tf, species) fraction in each cell type
    rowsum = np.nansum(arr, axis=1)[:, None]
    rowsum[rowsum == 0] = np.nan
    frac_per_row = arr / rowsum

    base = wide.index.to_frame(index=False)
    base['tau'] = tau_per_row
    base['argmax_ct'] = argmax_per_row
    frac_df = pd.DataFrame(frac_per_row, columns=cts, index=wide.index).reset_index(drop=True)
    base_with_frac = pd.concat([base, frac_df], axis=1)

    # Aggregate per (tf, cell_type). All fractions use TOTAL species in denominator
    # (not just species-with-data), so TFs with poor cross-species coverage rank lower.
    spec_rows = []
    by_tf = base_with_frac.groupby('tf_symbol', sort=False)
    for tf, sub in by_tf:
        n_total = sub.shape[0]
        n_with_data = int(sub['tau'].notna().sum())
        mean_tau = sub['tau'].mean(skipna=True)
        magnitude = sub[cts].mean(skipna=True)
        # raw argmax counts (NOT normalised); divide by n_total below
        argmax_counts = sub['argmax_ct'].value_counts(dropna=True)
        for ct in cts:
            spec_rows.append({
                'tf_symbol':    tf,
                'cell_type':    ct,
                'mean_tau':     mean_tau,
                'argmax_frac':  float(argmax_counts.get(ct, 0)) / n_total,
                'mean_frac_in': float(sub[ct].mean(skipna=True)),
                'mean_signal':  float(magnitude.get(ct, np.nan)),
                'n_species_with_data': n_with_data,
                'n_species_total':     n_total,
            })
    spec = pd.DataFrame(spec_rows)
    spec.to_parquet(OUT_PARQUET, index=False)
    print(f'\nWrote {OUT_PARQUET}  ({len(spec):,} rows)')

    # Top-10 per cell type, ranked by mean_frac_in (the fraction of the TF's
    # total cross-cell-type accessibility that lands in this cell type, averaged
    # across species). Filter to TFs with >=20/30 species of data.
    MIN_SPECIES_WITH_DATA = 20
    spec['rank_score'] = spec['mean_frac_in']
    spec_ok = spec[spec['n_species_with_data'] >= MIN_SPECIES_WITH_DATA]
    top10_rows = []
    for ct in cts:
        sub = spec_ok[spec_ok['cell_type'] == ct].copy()
        sub = sub.sort_values('rank_score', ascending=False).head(10)
        sub['rank'] = range(1, len(sub) + 1)
        top10_rows.append(sub)
    top10 = pd.concat(top10_rows, ignore_index=True)
    top10 = top10[['cell_type', 'rank', 'tf_symbol', 'rank_score', 'argmax_frac',
                   'mean_tau', 'mean_frac_in', 'mean_signal',
                   'n_species_with_data', 'n_species_total']]
    top10.to_csv(OUT_TOP10, sep='\t', index=False, float_format='%.4f')
    print(f'Wrote {OUT_TOP10}')
    print('\nSample (top-3 per cell type for first 3 cell types):')
    print(top10[top10['cell_type'].isin(cts[:3]) & (top10['rank'] <= 3)].to_string(index=False))


if __name__ == '__main__':
    main()
