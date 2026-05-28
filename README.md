# Cell-Aware miRNA Functional Repression Prediction

Final Year Project (BSc Biomedical Statistics, Xi'an Jiaotong-Liverpool University).
Three-axis hold-out diagnosis of the UTR-encoder shortcut in cell-type-aware
miRNA target prediction.

## Repository layout

- `script/` — training, evaluation, and plotting scripts
- `dataset/` — preprocessed parquet chunks (26 chunks × 4 files: samples,
  miRNA dict, UTR dict, cell dict; ~200 MB total)
- `cache/` — intermediate cache derived from `dataset/`: train/val/test
  splits, TargetScan context+ features, seed-anchored windows (~52 MB)
- `Fig*.svg` — schematic figure sources

## Quick start

Train the four models (cache is already included in this repo, will be reused):

```bash
./script/run_all.sh                       # CNN baseline (full)
./script/run_mixed_only.sh                # CNN baseline (mixed-only)
./script/run_v4.sh                        # ConvNeXt extended (full)
MIXED_ONLY=1 ./script/run_v4.sh           # ConvNeXt extended (mixed-only)
```

TargetScan-LR baseline (~3 min):

```bash
python script/train_targetscan_lr_baseline.py
```

Reproduce figures:

```bash
python script/prepare_fig4_data.py
python script/plot_fig4_dataset.py
python script/plot_fig42_comparison.py
python script/plot_fig43_per_cell_heatmap.py \
  --report runs/main_film_seed42/final_report.json \
  --title "CNN baseline (full)" \
  --out_name fig43a_cnn_baseline_per_cell
python script/plot_fig43_per_cell_heatmap.py \
  --report runs/v4_film_seed42/final_report.json \
  --title "ConvNeXt extended (full)" \
  --out_name fig43b_convnext_extended_per_cell
python script/plot_threshold_sweep.py
```

## Data preprocessing

The parquet chunks in `dataset/` were derived from the following public sources:

- scRNA-seq: GEO accession GSE138266 (PBMC, 16 immune cell types)
- miRNA-target annotations: miRTarBase release 9 (2025 update)
- miRNA sequences: miRBase release 22
- 3′ UTR sequences: Ensembl GRCh38 release 115
- miRNA activity inference: miTEA-HiRes (Lukasse et al., 2024)

Preprocessing pipeline (run once, snapshot included in this repo):

1. miTEA-HiRes activity inference per (miRNA, cell) pair
2. Rank-aware binary labelling (positive: `rank_percentile_low < 0.20`;
   negative: `rank_percentile_low > 0.50`)
3. Cross-product with miRTarBase MTI annotations
4. Chunked output as 26 parquet shards

The included parquet snapshot allows direct training without re-running the
upstream activity inference step.

## License

MIT
