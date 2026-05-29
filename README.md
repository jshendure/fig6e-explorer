# STEAM-v1 cross-species enhancer view (Fig 6e explorer)

Interactive recreation of **Fig 6e** from Shendure et al.,
*[evolutionary transfer learning](https://shendure.curve.space/articles/evolutionary-transfer-learning)*,
for arbitrary hg38 coordinates. Enter a position and a cell type; the app pulls the
hg38-projected STEAM-v1 predicted-accessibility track from all 239 Zoonomia mammals,
orders species by the Zoonomia 241-species phylogeny, and renders a viridis heatmap
with synteny-coverage and summed-strength tracks above.

## Layout

- **Top-left:** STEAM-v1 score color bar.
- **Top tracks:** per–100 bp bin, fraction of species with hg38 coverage (gray) and
  per-column normalized summed STEAM-v1 score across species (teal).
- **Left:** rectangular phylogram from the Zoonomia HAL guide tree, ladderized,
  with italic abbreviated tip labels (red for the `Highlight species` list).
- **Main:** viridis heatmap of predicted accessibility per species × 100 bp bin,
  NA cells set to the viridis floor (`#440154`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first query for a fresh locus fetches ~239 remote bigwigs (~30–60 s with 32
parallel workers); repeats are instant from the in-memory cache.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **New app** → pick this repo + branch + `app.py` → Deploy.

The free tier (Community Cloud) has 1 GB RAM, which fits ~20 cached loci. No secrets
or extra config are required; everything the app needs is fetched on demand.

## Data sources

- Per-species hg38-projected STEAM-v1 tracks:
  `https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/jax_atac_augmented_241_mammals_hg38/`
- Per-species native-coord enhancer BEDs (not used here; available at
  `.../jax_atac_augmented_241_mammals_core_enhancer/`).
- Zoonomia 241-mammal phylogeny (`data/zoonomia_241.nwk`): extracted via
  `halStats --tree` from the `241-mammalian-2020v2.hal` Cactus alignment. All 239
  species in the per-species set match a tree leaf exactly.

## Caveats

- The synteny track here is a **coverage proxy** from the hg38 projection, not a
  formal chain/HAL-derived span. Swap in real spans by editing
  `fig6e_core.compute_synteny` if you have a chain-based table.
- `Mus_musculus` and `Homo_sapiens` are the model references and are **not in the
  239-species per-species set**; if you highlight them they're silently ignored.
- Fig 6d (synteny network) and Fig 6f (per-cluster stratification) need the master
  enhancer-call synteny graph and are not implemented here.

## Project layout

```
.
├── app.py                  # Streamlit UI
├── fig6e_core.py           # core fetch / synteny / tree / plot (no Streamlit dep)
├── build_notebook.py       # builds fig6e_interactive.ipynb (notebook variant)
├── fig6e_interactive.ipynb # standalone notebook variant of the same logic
├── data/
│   └── zoonomia_241.nwk    # Zoonomia 241-mammal tree (Newick, ~12 KB)
├── requirements.txt
└── .streamlit/config.toml
```
