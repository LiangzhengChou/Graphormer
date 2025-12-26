# CIF Hardness Workflow (CSV + CIF)

This workflow adds a CIF-based hardness regression dataset that plugs into the
existing Graphormer training pipeline via the `graph_prediction` task.

## 1) CSV format

Prepare a CSV with **three required columns**:

```csv
id,cif_path,hardness
sample-001,./cifs/sample-001.cif,23.4
sample-002,./cifs/sample-002.cif,18.9
```

* `id`: unique identifier (string).
* `cif_path`: path to the CIF file. Relative paths are resolved relative to the
  CSV location.
* `hardness`: numeric target (regression).

Optional:

* `split`: `train`, `valid` (or `val`), or `test`. If any row includes `split`,
  then **all rows** must include it and each split must be non-empty.

If you want to generate splits automatically, use the helper script:

```bash
python examples/customized_dataset/prepare_cif_hardness_csv.py \
  --input-csv /path/to/hardness.csv \
  --output-csv /path/to/hardness_with_split.csv \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

## 2) Install CIF dependency

The dataset uses `pymatgen` to parse CIF files:

```bash
pip install pymatgen
```

## 3) Configure dataset environment variables

The dataset module reads configuration from environment variables:

```bash
export CIF_HARDNESS_CSV=/path/to/hardness.csv
export CIF_HARDNESS_CUTOFF=6.0
export CIF_HARDNESS_MAX_NEIGHBORS=32
export CIF_HARDNESS_DISTANCE_BINS=2.0,3.0,4.0,5.0
```

Notes:
* `CIF_HARDNESS_CSV` is **required**.
* `CIF_HARDNESS_CUTOFF` controls the neighbor search radius (Å).
* `CIF_HARDNESS_MAX_NEIGHBORS` caps neighbor count per atom (0 disables the cap).
* `CIF_HARDNESS_DISTANCE_BINS` controls distance bucketing for edge features.
  Values must be **strictly increasing**.

## 4) Train

Use the customized dataset module with `graph_prediction`:

```bash
fairseq-train \
  --user-dir ../../graphormer \
  --task graph_prediction \
  --user-data-dir examples/customized_dataset \
  --dataset-name customized_cif_hardness \
  --criterion l1_loss \
  --arch graphormer_base \
  --num-classes 1 \
  --batch-size 32 \
  --max-epoch 100 \
  --save-dir ./ckpts
```

The dataset will be split automatically into train/valid/test (80/10/10) using
the same split logic as other customized dataset examples.
If your CSV includes a `split` column, those values will be used instead.

## 5) Export predictions to CSV

After training, export predictions for train/valid/test using the checkpoint:

```bash
python examples/customized_dataset/export_hardness_predictions.py \
  --user-dir ../../graphormer \
  --task graph_prediction \
  --user-data-dir examples/customized_dataset \
  --dataset-name customized_cif_hardness \
  --criterion l1_loss \
  --arch graphormer_base \
  --num-classes 1 \
  --checkpoint-path /path/to/ckpts/checkpoint_best.pt \
  --output-dir ./predictions
```

This produces:

```
predictions/train_predictions.csv
predictions/valid_predictions.csv
predictions/test_predictions.csv
```

Each CSV contains: `id`, `true`, `prediction`.
