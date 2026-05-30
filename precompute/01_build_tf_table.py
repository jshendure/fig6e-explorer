"""Build a TF symbol -> hg38 TSS table from the Lambert 2018 list.

Sources:
  - http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt
    (one symbol per line; ~1639 high-confidence human TFs)
  - Ensembl REST bulk lookup endpoint for symbol -> gene record
    (POST https://rest.ensembl.org/lookup/symbol/homo_sapiens, max 1000/call)

Output: data/tfs_hg38.tsv with columns
  gene_symbol, ensembl_id, chrom, tss, strand, biotype, start, end

Symbols that don't resolve are written to data/tfs_unresolved.txt.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
OUT_TSV = REPO / 'data' / 'tfs_hg38.tsv'
UNRESOLVED_TXT = REPO / 'data' / 'tfs_unresolved.txt'

LAMBERT_URL = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt'
ENSEMBL_BULK = 'https://rest.ensembl.org/lookup/symbol/homo_sapiens'

VALID_CHROMS = {f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y']}


def fetch_tf_symbols() -> list[str]:
    r = requests.get(LAMBERT_URL, timeout=60)
    r.raise_for_status()
    syms = [line.strip() for line in r.text.splitlines() if line.strip()]
    print(f'Lambert 2018: {len(syms)} symbols.')
    return syms


def bulk_lookup(symbols: list[str], batch: int = 900) -> dict:
    out = {}
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        body = json.dumps({'symbols': chunk})
        for attempt in range(3):
            r = requests.post(ENSEMBL_BULK, data=body, headers=headers, timeout=120)
            if r.status_code == 429:  # rate limit
                wait = int(r.headers.get('Retry-After', '2'))
                print(f'  rate-limited, sleep {wait}s'); time.sleep(wait); continue
            r.raise_for_status()
            d = r.json()
            out.update(d)
            print(f'  {i + len(chunk)}/{len(symbols)} resolved (cumulative {len(out)}).')
            time.sleep(0.5)  # polite
            break
        else:
            raise RuntimeError(f'Failed to bulk-lookup chunk starting at {i}.')
    return out


def to_rows(symbols: list[str], lookups: dict) -> tuple[list[dict], list[str]]:
    rows, unresolved = [], []
    for sym in symbols:
        d = lookups.get(sym)
        if not d:
            unresolved.append(sym); continue
        seq = d.get('seq_region_name', '')
        chrom = seq if seq.startswith('chr') else f'chr{seq}'
        if chrom not in VALID_CHROMS:
            unresolved.append(f'{sym}\t{chrom}_excluded'); continue
        strand = d.get('strand')
        if strand not in (1, -1):
            unresolved.append(f'{sym}\tno_strand'); continue
        start = int(d['start']); end = int(d['end'])
        tss = start if strand == 1 else end
        rows.append({
            'gene_symbol': sym,
            'ensembl_id':  d.get('id'),
            'chrom':       chrom,
            'tss':         tss,
            'strand':      '+' if strand == 1 else '-',
            'biotype':     d.get('biotype'),
            'start':       start,
            'end':         end,
        })
    return rows, unresolved


def main():
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    symbols = fetch_tf_symbols()
    print('Bulk-looking up via Ensembl…')
    lookups = bulk_lookup(symbols)
    print(f'Got {len(lookups)} lookup hits.')
    rows, unresolved = to_rows(symbols, lookups)
    df = pd.DataFrame(rows).sort_values(['chrom', 'tss']).reset_index(drop=True)
    df.to_csv(OUT_TSV, sep='\t', index=False)
    if unresolved:
        UNRESOLVED_TXT.write_text('\n'.join(unresolved) + '\n')
    print(f'\nResolved {len(df)}/{len(symbols)} TFs -> {OUT_TSV}')
    if unresolved:
        print(f'Unresolved/excluded ({len(unresolved)}) -> {UNRESOLVED_TXT}')
    print('\nBy biotype:'); print(df['biotype'].value_counts().head())
    print('\nBy chrom:'); print(df['chrom'].value_counts().head())


if __name__ == '__main__':
    sys.exit(main())
