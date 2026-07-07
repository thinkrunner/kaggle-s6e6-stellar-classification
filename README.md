# Playground Series S6E6 - Stellar Classification (GALAXY / QSO / STAR)
**Kaggle Public Score: 0.96744**

---

## 🔗 Links
- [Kaggle Notebook](https://www.kaggle.com/code/thinkrunner/notebooked60bb2cd8)
- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e6)

---

## Project Summary

End-to-end ML pipeline for classifying SDSS (Sloan Digital Sky Survey) objects into `GALAXY`, `QSO`, and `STAR` using a 3-model ensemble.
Starting from a base model score of 0.95602, each step was carefully designed to improve Balanced Accuracy.

| Step | Technique | Score (OOF) |
|------|-----------|-------|
| Base model | First submission | 0.95602 |
| Sample Weight Tuning | Upweight confusable STAR samples | 0.96348 |
| K-Fold CV | StratifiedKFold(5) added | 0.96438 |
| Optuna + K-Fold | Per-model hyperparameter tuning | 0.96459 |
| 3-Model Ensemble | XGBoost + LightGBM + CatBoost | 0.96363 |
| Threshold Calibration | SciPy Nelder-Mead on class multipliers | 0.96658 |
| **Public LB (final submission)** | | **0.96744** |
| **Private LB (final submission)** | | **0.96685** |

---

## Pipeline Overview

```
Raw Data
  → Data Loading (train.csv / test.csv)
  → Feature Engineering (color indices, coordinate transform, interactions)
  → KBinsDiscretizer (fit on train only, transform on test → no leakage)
  → Label Encoding (target)
  → Feature Set Split (One-Hot for XGBoost / native categorical for LGB & CatBoost)
  → StratifiedKFold (5-Fold) Training
      → Sample Weighting (confusable STAR samples upweighted)
      → XGBoost + LightGBM + CatBoost (Optuna-tuned, early stopping)
  → OOF 3-Model Soft-Voting Ensemble
  → Threshold Calibration (SciPy Nelder-Mead, Balanced Accuracy objective)
  → Final Prediction & Submission
```

---

## Cross-Validation Strategy

- **Method**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Metric**: Balanced Accuracy (mean of per-class recall)
- **Leakage prevention**: `KBinsDiscretizer` fitted on train fold only; test set uses `transform()` exclusively

---

## Key Design Decisions

### 1. Feature Engineering
Derived features based on SDSS domain knowledge — filter differences (color indices) reflect a star's physical properties (temperature, spectral shape) far more reliably than raw magnitudes:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `u-g, g-r, r-i, i-z` | Adjacent-filter differences | Local spectral slope |
| `u-z, u-r, g-z` | Non-adjacent filter differences | Captures full UV–IR spectral trend (good for QSO's UV excess) |
| `log_redshift` | `log1p(max(0, redshift))` | Compresses QSO's long-tail redshift distribution for more even split points |
| `coord_x, coord_y, coord_z` | Spherical → 3D Cartesian from `alpha`/`delta` | Resolves the circularity issue of angular coordinates (0° = 360°) |
| `redshift_x_uz`, `redshift_x_ug`, `uz_x_ug` | redshift × color interactions | Explicitly exposes QSO's "high redshift + distinct color" combination |
| `photo_midrange` | `(max + min) / 2` across 5 filters | Robust summary of overall brightness level |
| `coord_{x,y,z}_500_bin` | Quantile-based 500-bin discretization | Approximates spatial clustering signal |

### 2. Sample Weighting
EDA showed STAR↔GALAXY confusion concentrated in specific `spectral_type × galaxy_population` combinations. Upweighted the corresponding STAR samples during training:
```python
sample_weight[(y_tr == star_label) & (galaxy_population == 'Blue_Cloud')] = 2.0
sample_weight[(y_tr == star_label) & (galaxy_population == 'Red_Sequence')] = 3.5
```

### 3. No Data Leakage
`KBinsDiscretizer` fitted only on train, then applied via `transform()` to test:
```python
kb = KBinsDiscretizer(n_bins=500, encode='ordinal', strategy='quantile', subsample=None)
df[bin_name] = kb.fit_transform(df[[col]]).ravel().astype(int)   # train
df[bin_name] = kb.transform(df[[col]]).ravel().astype(int)       # test (reuses train's bin edges)
```

### 4. 3-Model Ensemble
Soft-voting across three boosting algorithms, each tuned independently with Optuna:
```python
xgb_model = xgb.XGBClassifier(...)   # depth-wise boosting, GPU (Kaggle P100)
lgb_model = lgb.LGBMClassifier(...)  # leaf-wise boosting, local CPU (no Apple Silicon GPU support)
cat_model = CatBoostClassifier(...)  # ordered boosting, GPU (Kaggle P100)

ensemble_probs = (xgb_probs * 0.3333) + (lgb_probs * 0.3334) + (cat_probs * 0.3333)
```

### 5. Threshold Calibration
Balanced Accuracy is the mean of per-class recall, so multiplying OOF class probabilities by an optimal per-class weight can improve the score beyond plain argmax:
```python
def balanced_objective(weights):
    preds = np.argmax(oof_preds * weights, axis=1)
    return -balanced_accuracy_score(y_encoded, preds)

res = minimize(balanced_objective, initial_weights, method='Nelder-Mead')
```
This improved OOF Balanced Accuracy from 0.96363 → 0.96658 (+0.00294).

---

## Model Performance

| Model | Local CV (avg over 5 folds) | Kaggle Score | Gap |
|-------|---------------|--------------|-----|
| XGBoost (single) | 0.96306 | Not submitted individually | — |
| LightGBM (single) | 0.96328 | Not submitted individually | — |
| CatBoost (single) | 0.96290 | Not submitted individually | — |
| **3-Model Ensemble (uncalibrated)** | **0.96363** | — | — |
| **3-Model Ensemble + Threshold Calibration** | **0.96658** | **0.96744** | **0.00086** |

> **Gap analysis**: individual models (XGB/LGB/CAT) converged to nearly identical OOF scores (0.9629–0.9633), indicating low diversity in confusion patterns. As a result, the uncalibrated ensemble improved on the best single model by only ~0.0003–0.0007. Threshold calibration contributed a larger, more reliable gain (+0.00294) than the ensembling step itself.

---

## What Worked

- Sample weighting on confusable STAR samples → biggest single improvement (+0.00746 OOF)
- StratifiedKFold CV → more stable estimate and additional gain (+0.0009)
- Threshold calibration (Nelder-Mead on class multipliers) → consistent, low-cost improvement (+0.00294)
- No leakage in `KBinsDiscretizer` fit/transform split → reliable generalization from OOF to LB

## What Didn't Work / Limitations

- 3-model ensemble diversity → XGBoost, LightGBM, and CatBoost all learned nearly the same confusion pattern (same feature set), so ensembling gave only marginal gains over the best single model
- Interaction features (`redshift_x_uz`, `redshift_x_ug`, `uz_x_ug`) targeted the QSO axis (redshift), but the confusion identified in EDA (STAR↔GALAXY via `spectral_type × galaxy_population`) was not directly targeted by an equivalent feature
- Axis-independent quantile binning of `coord_x/y/z` (500 bins each) discretizes a spherical surface along separate axes, which can break true spatial adjacency near the poles/edges — not validated against a joint 3D clustering (e.g. KMeans) or HEALPix baseline

---

## Environment

```
Python      3.11.15
LightGBM    4.6.0
XGBoost     3.2.0
CatBoost    1.2.10
scikit-learn 1.8.0
scipy       1.16.3
optuna      4.8.0
```
