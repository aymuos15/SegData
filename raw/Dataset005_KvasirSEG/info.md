# Kvasir-SEG: Polyp Segmentation Dataset

## Overview

Kvasir-SEG is a dataset of gastrointestinal polyps from colonoscopy videos, annotated with binary segmentation masks. This is a 2D dataset focused on endoscopy polyp detection and segmentation.

## Dataset Details

- **Total Images**: 1000
- **Format**: JPEG images → converted to RGB PNG channels for nnU-Net
- **Masks**: Binary PNG (0 = background, 255 = polyp)
- **Image Resolution**: Variable (332×487 to 1920×1072)
- **Train/Test Split**: 80/20 (800 training, 200 test) applied by sorted filename

## Data Organization

```
kvasir-seg.zip (original)
├── images/          → JPEG files (*.jpg)
└── masks/           → Binary PNG masks
```

### Conversion to nnU-Net Format

1. Images are loaded as JPEG and converted to RGB
2. Each image is split into 3 separate PNG channels: `{case_id}_0000.png`, `{case_id}_0001.png`, `{case_id}_0002.png`
3. Masks are binarized with threshold 127: `(pixel_value > 127) → 0 or 255`
4. Case IDs are zero-padded 3-digit integers: `000`, `001`, ..., `999`
5. Cases `000-799` go to training split (imagesTr/labelsTr)
6. Cases `800-999` go to test split (imagesTs/labelsTs)

## Source Information

**Reference**: https://datasets.simula.no/kvasir-seg/

**Citation**:
Jha, D. et al. Kvasir-SEG: A Segmented Polyp Dataset. In: MultiMedia Modeling. MMM 2020. Lecture Notes in Computer Science, vol 11961. Springer. https://doi.org/10.1007/978-3-030-37734-2_37

**License**: Research/Educational use only
See https://datasets.simula.no/kvasir-seg/ for full license terms

## Additional Notes

- Bounding box JSON included in original zip is not used in this conversion
- All polyps are single-class (no distinction between polyp types)
- Some images may contain multiple polyps; connected component analysis counts individual polyps
- Images vary significantly in resolution; models should handle variable input sizes
