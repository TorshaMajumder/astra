# HATS Notebooks

Cross-match of Gaia DR3 variability catalogs, PS1 DR2, and Zubercal DR16, stored in [HATS](https://hats.readthedocs.io) format.

## Workflow

Run the notebooks in this order:

1. **`import-vizier.ipynb`** — downloads and converts the source catalogs from VizieR into HATS format.
2. **`zubercal-x-gaia-vars.ipynb`** — cross-matches the Zubercal DR16 catalog with Gaia variable star catalog.

## Loading examples

Notebooks prefixed with `_` show how to load the resulting HATS catalogs:

- `_load-with-lsdb.ipynb` — loading with [LSDB](https://docs.lsdb.io)
- `_load-with-nested-pandas.ipynb` — loading with [nested-pandas](https://nested-pandas.readthedocs.io)

## Results

The cross-matched catalog for the `vclassre` sample is published on Hugging Face:

https://huggingface.co/datasets/snad-space/astra-zubercaldr16_gaiadr3vclassre
