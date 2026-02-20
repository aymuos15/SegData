# Download MOTUM Dataset

## Steps

1. **Visit G-Node Repository** (recommended):
   Navigate to https://doi.gin.g-node.org/10.12751/g-node.tvzqc5/

   Alternative: Harvard Dataverse at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KUUEWC

2. **Accept License**:
   The dataset is licensed under CC BY 4.0. Review and accept the terms.

3. **Request Access** (if required):
   File access may require an account. Follow the on-screen instructions.

4. **Download**:
   Download the main dataset zip file (G-Node: `10.12751_g-node.tvzqc5.zip` or similar).

5. **Place in Directory**:
   Move the downloaded zip file to this directory (`raw/Dataset003_MOTUM/`).

6. **Run Setup**:
   ```bash
   python setup.py
   ```

This script will:
- Read the BIDS-formatted dataset (sub-XXXX/anat/ and derivatives/ directories)
- Find all patient directories
- Copy 4 MRI channels per patient
- Merge two segmentation labels into one 3-class mask
- Apply 80/20 train/test split (sorted by patient ID)
- Update `dataset.json` with actual counts

---

## Troubleshooting

- **"No .zip file found"**: Ensure the zip file is in this directory
- **"No NIfTI files found"**: Check that the extracted structure contains `.nii.gz` files with FLAIR image
- **Missing labels**: Some cases may have missing label files; the script will skip them

## Expected Output

```
imagesTr/      → ~54 training cases × 4 channels + 1 label
labelsTr/      → ~54 training segmentation masks
imagesTs/      → ~13 test cases × 4 channels + 1 label
labelsTs/      → ~13 test segmentation masks
dataset.json   → Updated metadata
```
