# Batch A3 Inference

This folder contains a standalone batch inference script for raw A3 image folders.

Expected default input layout:

```text
input_root/
  patient_001/
    A3_001.png
    A3_002.png
  patient_002/
    A3_001.png
```

By default, outputs are written beside each source image:

```text
patient_001/
  A3_001_pred_mask.png
  A3_001_pred_color.png
  A3_001_overlay.png
  A3_001_measurement.png
```

Run KnowSAM/SGDL inference:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --model-path ".\Results\train_260513_data_label1_v100_semi_106_117_13_13\SGDL_best_model.pth"
```

Write outputs to a separate root while preserving patient folders:

```bash
python batch_inference/batch_infer_a3.py ^
  --input-root "D:\A3_images" ^
  --output-root "D:\A3_predictions" ^
  --model-path ".\Results\train_260513_data_label1_v100_semi_106_117_13_13\SGDL_best_model.pth"
```

Run A3-PASS inference:

```bash
python batch_inference/batch_infer_a3.py ^
  --variant a3_pass ^
  --input-root "D:\A3_images" ^
  --model-path ".\Results\A3_PASS_KnowSAM_V100_label1_106_117_13_13\fold_0\PASS_best_model.pth"
```

Useful options:

- `--include-keyword A3`: only process filenames containing `A3`.
- `--device cpu` or `--device cuda:0`: override automatic device selection.
- `--save-prob`: also save foreground probability PNG files.
- `--pixel-spacing 0.12` or `--pixel-spacing 0.12,0.12`: also report width/depth in mm.
- `--disable-measurement`: skip fissure width/depth measurement and `*_measurement.png`.
- `--overwrite`: overwrite existing `*_pred_mask.png`, `*_overlay.png`, and `*_measurement.png`.

The measurement overlay draws dashed lines for:

- `width`: the opening distance between the two lips of the lateral fissure.
- `depth`: the perpendicular distance from the fissure sulcus bottom to the opening line.

Numeric fields are written to `batch_inference_summary.csv` as `fissure_width_px`, `fissure_depth_px`, `fissure_mean_width_px`, and mm fields when pixel spacing is provided.

Measure saved masks directly:

```bash
python scripts/measure_output_masks.py ^
  --input-root "Results\data_260513" ^
  --output-dir "Results\data_260513\测量结果_按指标分类" ^
  --overwrite
```

The direct-mask measurement script saves each measurement type in its own folder:

```text
测量结果_按指标分类/
  外侧裂开口宽度/
    image_overlay/
    mask_overlay/
  外侧裂最大深度/
    image_overlay/
    mask_overlay/
  裂隙弯曲度/
    image_overlay/
    mask_overlay/
  角度/
    image_overlay/
    mask_overlay/
  纵裂分支最大深度/
    image_overlay/
    mask_overlay/
  纵裂全长/
    image_overlay/
    mask_overlay/
  纵裂面积/
    image_overlay/
    mask_overlay/
  tables/
    measurement_results.csv
    summary.json
  logs/
    measure_output_masks.log
```

For multiclass masks, the direct script uses class `1` for lateral fissure metrics and class `2` for longitudinal fissure metrics by default. Override them with `--lateral-class` and `--longitudinal-class` if the label definition changes.

Angle measurement is only reported when both left and right lateral fissure components are present. By default, the reference line connects the pre-junction terminals of the left and right lateral fissure upper arcs: the terminal is the point just before the upper arc enters the branch core, horizontal branch, or lower arm. Branching masks use a branch exclusion zone (`--branch-cut-radius`, default `6` px) and a maximum terminal depth guard (`--terminal-max-depth-ratio`, default `0.62`). Non-branching arc masks use the superior portion of the main skeleton path (`--upper-arc-end-ratio`, default `0.55`). The apex is only displayed when available and does not set the reference line. For each side, the script fits a local PCA tangent using only points before the terminal (`--terminal-tangent-back-length`, default `28` px), then reports the acute angle between that tangent and the reference direction.
