# BTXRD: Bone Tumor X-ray Radiograph Dataset

## Overview

BTXRD is a multi-institution bone tumor X-ray radiograph dataset with COCO-style polygon segmentation annotations. The dataset contains 3,746 total images across three categories (benign, malignant, normal), but only the 1,867 tumor images (benign + malignant) have segmentation masks. Normal radiographs are excluded from this conversion.

## Dataset Details

- **Total Images (original)**: 3,746 (1,525 benign + 342 malignant + 1,879 normal)
- **Annotated Images**: 1,867 (benign + malignant only)
- **Format**: JPEG images + COCO JSON annotations -> converted to grayscale PNG for nnU-Net
- **Masks**: Binary PNG (0 = background, 255 = tumor)
- **Image Resolution**: Variable
- **Train/Test Split**: 80/20 (applied by sorted filename)

## Original Zip Structure

```
BTXRD.zip
├── dataset.csv           -> Labels: benign, malignant, normal
├── images/               -> JPEG radiograph files (IMG{id}.jpeg)
└── Annotations/          -> COCO-style JSON files (IMG{id}.json)
```

The zip may contain a nested directory (e.g., `BTXRD/`) inside.

## Annotation Format

Each JSON annotation file follows COCO format:
```json
{
  "images": [{"id": 1, "width": W, "height": H, "file_name": "..."}],
  "annotations": [
    {"id": 1, "image_id": 1, "segmentation": [[x1, y1, x2, y2, ...]], ...},
    ...
  ]
}
```

- `images[0].width` / `images[0].height`: image dimensions
- `annotations[].segmentation`: list of polygon regions, each a flat list `[x1, y1, x2, y2, ...]`
- Multiple annotations per image are merged into a single binary mask
- Polygons with fewer than 6 coordinates (3 points) are skipped as degenerate

## Conversion to nnU-Net Format

1. `dataset.csv` is read to identify non-normal images (Label != 'normal')
2. For each annotated tumor image:
   - Image dimensions are read from the COCO JSON (`images[0].width/height`)
   - Falls back to `Image.open().size` if JSON lacks dimensions
   - JPEG images are converted to grayscale PNG (`Image.convert('L')`)
   - COCO polygon annotations are rasterized to binary masks using `PIL.ImageDraw.polygon`
3. Cases are sorted by filename, then split 80/20 into train/test
4. Case IDs are sequential zero-padded integers: `000`, `001`, ...

## Source Information

**Reference**: https://doi.org/10.6084/m9.figshare.27865398

**Citation**:
Bhatt, P. et al. A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors. Scientific Data 12, 2025. https://doi.org/10.1038/s41597-024-04311-y

**License**: CC BY 4.0

## Additional Notes

- Normal radiographs have no segmentation annotations and are excluded
- Some tumor images may have multiple annotated regions; all are merged into one binary mask
- Image filenames follow the pattern `IMG{id}.jpeg` (or `.jpg`)
- Annotation filenames follow the pattern `IMG{id}.json`
