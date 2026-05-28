# Cell-Aware miRNA Functional Repression Prediction

Final Year Project (BSc Biomedical Statistics, Xi'an Jiaotong-Liverpool University).
Three-axis hold-out diagnosis of the UTR-encoder shortcut in cell-type-aware
miRNA target prediction.

## Repository layout
- `script/` — training, evaluation, and plotting scripts
- `Fig*.svg` — schematic figure sources

## Quick start

```bash
# 1. Build cache from raw data (GSE138266 + miRTarBase 2025 + Ensembl + miRBase)
./script/run_all.sh

# 2. Train the four models
./script/run_all.sh                       # CNN baseline (full)
./script/run_mixed_only.sh                # CNN baseline (mixed-only)
./script/run_v4.sh                        # ConvNeXt extended (full)
MIXED_ONLY=1 ./script/run_v4.sh           # ConvNeXt extended (mixed-only)

# 3. TargetScan-LR baseline (~3 min)
python script/train_targetscan_lr_baseline.py

# 4. Reproduce figures
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

## Data sources
- scRNA-seq: GEO accession GSE138266
- miRNA-target annotations: miRTarBase release 9 (2025)
- miRNA sequences: miRBase release 22
- UTR sequences: Ensembl GRCh38 release 115
- Activity inference: miTEA-HiRes

Processed cache (~ several GB) is not included; rebuild from raw sources
using the scripts above.

## License
MIT
