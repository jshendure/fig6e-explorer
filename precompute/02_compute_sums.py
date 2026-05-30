"""Bulk-fetch sum + mean of predicted accessibility in ±50 kb windows around
each TF TSS, across 30 phylogenetically-spread Zoonomia species × 32 cell types.

Inputs:
  - data/tfs_hg38.tsv (from 01_build_tf_table.py)
  - data/zoonomia_241.nwk
  - hg38-projected bigwigs at
    https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/
        jax_atac_augmented_241_mammals_hg38/hg38/{species}/{species}.{cell_type}.bw

Outputs:
  - data/tf_window_sums.parquet  (long; tf_symbol, species, cell_type, sum, mean,
                                  window_bp, chrom_clipped)
  - cache/tf_sums_partial/{species}__{cell_type}.parquet  (per-task partials; idempotent)

Parallelism: 48 worker threads, each opening one bigwig and doing 1622 sum+mean
stats calls (~3.2 k calls/bigwig). Expected wall time ~15-25 min.
"""
from __future__ import annotations
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fig6e_core as core
import pyBigWig

REPO = Path(__file__).resolve().parent.parent
TF_TSV = REPO / 'data' / 'tfs_hg38.tsv'
OUT_PARQUET = REPO / 'data' / 'tf_window_sums.parquet'
PARTIAL_DIR = REPO / 'cache' / 'tf_sums_partial'
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_KB = 50
N_SPECIES = 30
MAX_WORKERS = 48

# Hard-coded so we don't pay an extra HTTP round trip per species to read chrom dict.
HG38_CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415,
}


def species_subset() -> list[str]:
    all_sp = set(core.list_species())
    ordered = core.tree_leaf_order(restrict_to=all_sp)
    return core.subsample_evenly(ordered, N_SPECIES)


def fetch_one_bw(species: str, cell_type: str, tf_df: pd.DataFrame,
                 window_bp: int) -> pd.DataFrame:
    """Open one bigwig, query window sum+mean for every TF, return long df."""
    partial = PARTIAL_DIR / f'{species}__{cell_type}.parquet'
    if partial.exists():
        return pd.read_parquet(partial)

    url = core.HG38_BW_FMT.format(species=species, cell_type=cell_type)
    rows = []
    try:
        bw = pyBigWig.open(url)
    except Exception as e:
        print(f'  [open fail] {species}/{cell_type}: {e}', flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(partial, index=False)
        return df

    try:
        for _, t in tf_df.iterrows():
            chrom = t['chrom']
            chrom_size = HG38_CHROM_SIZES.get(chrom)
            if chrom_size is None:
                continue
            start = max(0, int(t['tss']) - window_bp)
            end = min(chrom_size, int(t['tss']) + window_bp)
            if end <= start:
                continue
            try:
                s = bw.stats(chrom, start, end, type='sum', nBins=1)[0]
                m = bw.stats(chrom, start, end, type='mean', nBins=1)[0]
            except Exception:
                s, m = None, None
            rows.append({
                'tf_symbol': t['gene_symbol'],
                'species': species,
                'cell_type': cell_type,
                'sum': float(s) if s is not None else np.nan,
                'mean': float(m) if m is not None else np.nan,
                'window_bp': end - start,
                'chrom_clipped': (end - start) != 2 * window_bp,
            })
    finally:
        bw.close()

    df = pd.DataFrame(rows)
    df.to_parquet(partial, index=False)
    return df


def main():
    if not TF_TSV.exists():
        sys.exit(f'Missing {TF_TSV}; run 01_build_tf_table.py first.')
    tf_df = pd.read_csv(TF_TSV, sep='\t')
    print(f'{len(tf_df)} TFs loaded.')

    species = species_subset()
    print(f'{len(species)} species selected (phylogenetically subsampled).')
    print(f'{len(core.CELL_TYPES)} cell types.')

    tasks = [(sp, ct) for sp in species for ct in core.CELL_TYPES]
    print(f'{len(tasks)} (species, cell_type) bigwig fetches to run.')

    window_bp = WINDOW_KB * 1000
    t0 = time.time()
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one_bw, sp, ct, tf_df, window_bp): (sp, ct)
                for sp, ct in tasks}
        for fut in as_completed(futs):
            sp, ct = futs[fut]
            try:
                df = fut.result()
            except Exception as e:
                failed += 1
                print(f'  [err] {sp}/{ct}: {e}', flush=True)
                continue
            done += 1
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate else 0
                print(f'  {done}/{len(tasks)} done in {elapsed:.0f}s '
                      f'({rate*60:.1f}/min, ETA {eta:.0f}s)', flush=True)

    print(f'\nMerging partials -> {OUT_PARQUET}')
    parts = sorted(PARTIAL_DIR.glob('*.parquet'))
    df_all = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df_all.to_parquet(OUT_PARQUET, index=False)
    print(f'  {len(df_all):,} rows; ~{OUT_PARQUET.stat().st_size/1e6:.1f} MB')
    print(f'  TFs: {df_all["tf_symbol"].nunique()}, '
          f'species: {df_all["species"].nunique()}, '
          f'cell types: {df_all["cell_type"].nunique()}')
    print(f'Total: {time.time() - t0:.0f}s')
    if failed:
        print(f'WARNING: {failed} tasks failed.')


if __name__ == '__main__':
    main()
