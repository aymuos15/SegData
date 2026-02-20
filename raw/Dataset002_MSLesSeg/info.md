# MSLesSeg Dataset Information

## Format

**3D volumetric NIfTI FLAIR format** (not 2D slices)
- File extension: `.nii.gz`
- Dimensions: 182 × 218 × 182 voxels
- Spacing: 1mm isotropic
- Orientation: LAS (Left-Anterior-Superior)
- Registered to: MNI152 template space
- Single sequence: FLAIR only (T1 and T2 not used)

## Channel

Each case has 1 FLAIR MRI sequence:

| Channel | Sequence | Filename |
|---------|----------|----------|
| 0       | FLAIR    | `{case_id}_0000.nii.gz` |

## Labels

Binary semantic segmentation:
- **0**: Background (healthy tissue)
- **1**: White matter lesion (MS pathology)

**Note**: Labels are already binary (no instance-to-semantic conversion needed, unlike Dataset001_Cellpose)

## Preprocessing

Data is **already preprocessed**:
- Brain extraction (skull removed)
- Registration to MNI152 1mm template
- Interpolated to 182×218×182 voxels
- Cropped to brain bounding box

## Data Split

115 total scan series from 75 patients:
- 50 RRMS (Relapsing-Remitting MS)
- 5 PPMS (Primary Progressive MS)
- ~67% training, ~33% test (split determined by setup.py based on downloaded structure)

## Citation

Guarnera, F., Rondinella, A., Crispino, E. et al. (2025). MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset. **Scientific Data**, 12, 920.
https://doi.org/10.1038/s41597-025-05250-y
