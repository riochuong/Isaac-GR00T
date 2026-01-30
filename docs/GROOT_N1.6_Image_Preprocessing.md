# GR00T N1.6 Image Preprocessing Pipeline

This document explains how GR00T N1.6 processes images from cameras with different resolutions.

## Example Dataset Configuration

| Camera | Resolution | Aspect Ratio |
|--------|------------|--------------|
| Scene | 848×480 | 1.77:1 (wide) |
| Wrist | 640×480 | 1.33:1 (4:3) |

## Default Processing Settings

From `gr00t/configs/model/gr00t_n1d6.py`:

```python
shortest_image_edge: int = 256
crop_fraction: float = 0.95
use_albumentations_transforms: bool = True
```

---

## Why 256 as the Shortest Edge?

### The Vision Encoder Architecture

From `Eagle-Block2A-2B-v2/config.json`:

```json
{
  "vision_config": {
    "patch_size": 14,
    "model_type": "siglip2_vision_model"
  },
  "use_pixel_shuffle": true,
  "pixels_per_token": 784
}
```

### How It Works

1. **SigLIP-2 Vision Encoder** divides the image into 14×14 pixel patches
2. **Pixel Shuffle (2×2)** combines 4 adjacent patches into 1 token
3. **Effective token size**: 28×28 = 784 pixels per token

### Token Count for Different Image Sizes

| Image Size | Patches | Tokens (after pixel shuffle) |
|------------|---------|------------------------------|
| 224×224 | 16×16 = 256 | 64 tokens |
| **256×256** | 18×18 = 324 | **~81 tokens** |
| 384×384 | 27×27 = 729 | 169 tokens |
| 448×448 | 32×32 = 1024 | 256 tokens |
| 512×512 | 36×36 = 1296 | 324 tokens |

### Why 256 is the Default

1. **Efficiency vs Quality Tradeoff**:
   - 256 → ~81 tokens per image (manageable for training)
   - 448 → ~256 tokens (4× more compute/memory)
   - Robot manipulation doesn't need 4K resolution!

2. **Sufficient for Manipulation Tasks**:
   - Gripper, objects, workspace clearly visible at 256px
   - Fine details (screws, wires) may need higher resolution

3. **Matches Common VLM Practice**:
   - CLIP: 224×224
   - SigLIP: 224, 256, 384
   - LLaVA: 336×336
   - GR00T default: 256

4. **Configurable**:
   ```python
   # In your config, you can increase resolution:
   config.model.shortest_image_edge = 384  # More detail, more compute
   config.model.shortest_image_edge = 448  # Even more detail
   ```
   - Tradeoff: more tokens = more memory/compute = potentially better detail

---

## Image Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMAGE PREPROCESSING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Scene (848×480)              Wrist (640×480)                       │
│       │                            │                                 │
│       ▼                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. SmallestMaxSize(256) - Scale so shortest edge = 256      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                            │                                 │
│       ▼                            ▼                                 │
│   452×256                      341×256                              │
│       │                            │                                 │
│       ▼                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. FractionalCrop(0.95) - Random/Center crop 95%            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                            │                                 │
│       ▼                            ▼                                 │
│   429×243                      323×243                              │
│       │                            │                                 │
│       ▼                            ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 3. SmallestMaxSize(256) - Resize back                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                            │                                 │
│       ▼                            ▼                                 │
│   ~451×256                     ~340×256                             │
│   (147 tokens)                 (111 tokens)                         │
│       │                            │                                 │
│       └────────────┬───────────────┘                                │
│                    ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Eagle VLM (SigLIP-2 Vision Encoder)                         │   │
│  │ - Handles variable-length token sequences natively          │   │
│  │ - No forced square resize needed!                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Understanding `SmallestMaxSize` - Aspect Ratio Preservation

`SmallestMaxSize` is an Albumentations transform that scales an image so that its **smallest dimension** becomes the target size, while maintaining the aspect ratio.

### The Math

```
Given:
  - Original image: W × H
  - Target max_size: 256

Scale factor = max_size / min(W, H)

New dimensions:
  - new_W = W × scale_factor
  - new_H = H × scale_factor
```

### Example Calculations

#### Scene Camera: 848×480

```
Original aspect ratio: 848/480 = 1.7667

Smallest dimension: 480
Scale factor: 256 / 480 = 0.5333

New dimensions:
  Width:  848 × 0.5333 = 452
  Height: 480 × 0.5333 = 256

New aspect ratio: 452/256 = 1.7667 ✓ (preserved!)
```

#### Wrist Camera: 640×480

```
Original aspect ratio: 640/480 = 1.3333

Smallest dimension: 480
Scale factor: 256 / 480 = 0.5333

New dimensions:
  Width:  640 × 0.5333 = 341
  Height: 480 × 0.5333 = 256

New aspect ratio: 341/256 = 1.3333 ✓ (preserved!)
```

### Why Aspect Ratio is Preserved

**SmallestMaxSize scales BOTH dimensions by the SAME factor:**

```
new_W / new_H = (W × scale) / (H × scale) = W / H  ✓
```

### Comparison: SmallestMaxSize vs Resize(256, 256)

| Method | Scene Result | Wrist Result | Aspect Ratio |
|--------|--------------|--------------|--------------|
| `SmallestMaxSize(256)` | 452×256 | 341×256 | **Preserved** |
| `Resize(256, 256)` | 256×256 | 256×256 | **Distorted** |

**Visual Comparison:**

```
SmallestMaxSize(256) - ASPECT RATIO PRESERVED

Scene Camera (848×480 → 452×256):
┌──────────────────────────────────────────┐
│                                          │
│            Wide rectangle                │
│              stays wide!                 │
│                                          │
└──────────────────────────────────────────┘

Wrist Camera (640×480 → 341×256):
┌──────────────────────────────────┐
│                                  │
│         4:3 rectangle            │
│           stays 4:3!             │
│                                  │
└──────────────────────────────────┘

----------------------------------------------------------------------

Resize(256, 256) - ASPECT RATIO DESTROYED

Scene Camera (848×480 → 256×256):
┌─────────────────────────┐
│                         │
│   Wide scene SQUISHED   │
│   Objects look tall     │
│      and thin!          │
└─────────────────────────┘

Wrist Camera (640×480 → 256×256):
┌─────────────────────────┐
│                         │
│   4:3 image SQUISHED!   │
│   Circles become ovals  │
│                         │
└─────────────────────────┘
```

---

## Dynamic Token Calculation

From `processor_config.json`:
```json
{
  "pixels_per_token": 784
}
```

**Token count formula:**
```
image_tokens = (height × width) / pixels_per_token
```

### Your Cameras → Token Counts

| Camera | After Preprocessing | Pixels | Tokens |
|--------|---------------------|--------|--------|
| Scene | 451×256 | 115,456 | **147** |
| Wrist | 340×256 | 87,040 | **111** |

**Key insight:** Different cameras produce different numbers of tokens, which is handled natively by the Transformer architecture.

---

## Why Different Resolutions Work with GR00T N1.6

1. **Aspect Ratio Preservation**: `SmallestMaxSize` scales images while preserving aspect ratio (no distortion)

2. **Dynamic Token Count**: Eagle VLM calculates tokens dynamically based on actual image dimensions

3. **Native Resolution Support**: Unlike older VLMs that force 224×224, Eagle/SigLIP-2 supports flexible aspect ratios

4. **Consistent Augmentation**: `ReplayCompose` ensures the same random augmentation is applied to all temporal frames of the same camera

---

## Code References

### Image Transformations (Albumentations)

From `gr00t/model/gr00t_n1d6/image_augmentations.py`:

```python
def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    # Training transforms
    train_transform_list = [
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]
    
    # Optional augmentations
    if random_rotation_angle:
        train_transform_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))
    if color_jitter_params:
        train_transform_list.append(A.ColorJitter(...))
    
    train_transform = A.ReplayCompose(train_transform_list, p=1.0)
    
    # Evaluation transforms (deterministic)
    eval_transform = A.Compose([
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalCenterCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ])
    
    return train_transform, eval_transform
```

### Token Calculation in Eagle VLM

From `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/processing_eagle3_vl.py`:

```python
# Dynamic token calculation based on actual image size
image_inputs = self.image_processor(images=[image], ...)
image_height, image_width = image_inputs['image_sizes'][0]
image_tokens = image_height * image_width // self.pixels_per_token
```

---

## Summary

**Your dataset with different camera resolutions (848×480 and 640×480) will work out of the box with GR00T N1.6.** No extra preprocessing steps are required because:

1. The pipeline preserves aspect ratios
2. Token counts are calculated dynamically
3. The Transformer architecture handles variable-length sequences natively
