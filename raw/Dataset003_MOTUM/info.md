# MOTUM Brain Tumor MRI Dataset

## Overview

The MOTUM (Multi-center, Multi-origin brain Tumor MRI) dataset consists of 67 patients with gliomas and metastases from multiple centers. Each patient has four MRI sequences acquired:

- **FLAIR**: Fluid-Attenuated Inversion Recovery
- **T1**: T1-weighted imaging
- **T2**: T2-weighted imaging
- **T1ce**: T1-weighted with contrast enhancement (gadolinium)

---

## Segmentation Labels

The raw dataset provides **two separate binary segmentation masks per patient**:

1. **`flair_seg_label1.nii.gz`**: Tumor region visible on FLAIR (FLAIR-visible tumor)
2. **`t1ce_seg_label2.nii.gz`**: Tumor region visible on T1ce (contrast-enhancing tumor)

### Label Merging Strategy

These are merged into a **single 3-class segmentation mask** (`{case_id}.nii.gz`):

| Value | Region | Source | Priority |
|-------|--------|--------|----------|
| 0 | Background | N/A | N/A |
| 1 | FLAIR-only tumor | `flair_seg_label1=1` AND `t1ce_seg_label2=0` | — |
| 2 | T1ce-enhancing tumor | `t1ce_seg_label2=1` | **High** (takes priority in overlaps) |

**Merging logic in NumPy**:
```python
label_merged = np.zeros_like(label1)
label_merged[label1 == 1] = 1
label_merged[label2 == 1] = 2  # Overwrites label 1 in overlapping regions
```

---

## Train/Test Split

The dataset provides **no predetermined train/test split**. Cases are split **80/20 by index** after sorting by patient ID:

- **Training**: 54 cases (indices 0–53)
- **Test**: 13 cases (indices 54–66)

Splitting by ID order ensures reproducibility and prevents data leakage within multi-center cohorts.

---

## nnU-Net Format

Cases are converted to nnU-Net format with the following structure:

```
Dataset003_MOTUM/
├── imagesTr/
│   ├── 000_0000.nii.gz  (FLAIR)
│   ├── 000_0001.nii.gz  (T1)
│   ├── 000_0002.nii.gz  (T2)
│   ├── 000_0003.nii.gz  (T1ce)
│   ├── 001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── 000.nii.gz       (3-class label)
│   ├── 001.nii.gz
│   └── ...
├── imagesTs/
│   ├── 000_0000.nii.gz
│   └── ...
└── labelsTs/
    ├── 000.nii.gz
    └── ...
```

Channel indices (`_0000`, `_0001`, etc.) correspond to:
- `_0000`: FLAIR
- `_0001`: T1
- `_0002`: T2
- `_0003`: T1ce

---

## Statistics

- **Total cases**: 67
- **Training**: 54
- **Test**: 13
- **Channels per case**: 4 (FLAIR, T1, T2, T1ce)
- **Label classes**: 3 (background, FLAIR-only tumor, T1ce-enhancing tumor)
- **Spacing**: Native (not resampled; varies by acquisition center)
- **Format**: NIfTI (`.nii.gz`)

---

## References

- **Dataset**: Gong Z. et al. A Multi-Center, Multi-Parametric MRI Dataset of Primary and Secondary Brain Tumors. Harvard Dataverse (2023). doi:10.7910/DVN/KUUEWC
- **GitHub**: https://github.com/hongweilibran/MOTUM
- **License**: CC BY 4.0

---

## Notes for Practitioners

1. **Multi-center**: Data from multiple imaging centers with potential acquisition variation.
2. **No resampling**: Voxel spacing is heterogeneous; models should handle variable dimensions or pre-processing pipelines should resample to standard space.
3. **Label overlap**: T1ce regions may semantically represent more active/enhancing tumor. Class 2 takes priority in the merged mask.
4. **No predetermined split**: Train/test split is derived algorithmically (80/20 by ID) for reproducibility.
