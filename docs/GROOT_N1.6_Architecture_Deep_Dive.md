# GR00T N1.6: Complete Architecture Deep Dive

This document provides a comprehensive explanation of the GR00T N1.6 Vision-Language-Action (VLA) model architecture, including training and inference pipelines, tokenization, embeddings, and implementation details.

## Table of Contents

1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Detailed Architecture Visualization](#detailed-architecture-visualization)
3. [Key Components Breakdown](#key-components-breakdown)
4. [Tokenization & Embeddings](#tokenization--embeddings)
5. [Model Configuration](#model-configuration)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Key Implementation Files](#key-implementation-files)
9. [N1.6 vs N1.5 Comparison](#key-architectural-decisions-in-n16-vs-n15)
10. [Flow Matching Explained](#flow-matching-vs-standard-diffusion)

---

## High-Level Architecture Overview

GR00T N1.6 is a **Vision-Language-Action (VLA)** model that combines:
1. **System 2 (Vision-Language Model)**: Eagle VLM backbone for understanding images and language
2. **System 1 (Diffusion Transformer)**: Flow-matching diffusion head for action generation

Here's the official architecture diagram from the codebase:

![Architecture](../media/model-architecture.png)

---

## Detailed Architecture Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           GR00T N1.6 COMPLETE ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              INPUT PROCESSING PIPELINE                               │   │
│  ├──────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                      │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                  │   │
│  │  │  Image Input    │    │  Language Input │    │  Robot State    │                  │   │
│  │  │ (B,T,H,W,3)     │    │    "Pick up     │    │ (B,T_s,D_state) │                  │   │
│  │  │  uint8          │    │     the cup"    │    │   float32       │                  │   │
│  │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘                  │   │
│  │           │                      │                      │                           │   │
│  │           ▼                      ▼                      ▼                           │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                  │   │
│  │  │ Image Transform │    │   Tokenizer     │    │ State Normalizer│                  │   │
│  │  │ - Resize        │    │ (Eagle/Qwen)    │    │ - Min/Max norm  │                  │   │
│  │  │ - ColorJitter   │    │                 │    │ - Sin/Cos enc   │                  │   │
│  │  │ - RandomCrop    │    │                 │    │ [-1, 1] range   │                  │   │
│  │  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘                  │   │
│  │           │                      │                      │                           │   │
│  │           ▼                      ▼                      │                           │   │
│  │  ┌──────────────────────────────────────────┐           │                           │   │
│  │  │        Eagle VLM Processor               │           │                           │   │
│  │  │  - Image → Pixel Values                  │           │                           │   │
│  │  │  - Text → Input IDs + Attention Mask     │           │                           │   │
│  │  │  - Chat Template Formatting              │           │                           │   │
│  │  └────────────────────┬─────────────────────┘           │                           │   │
│  │                       │                                 │                           │   │
│  └───────────────────────┼─────────────────────────────────┼───────────────────────────┘   │
│                          │                                 │                               │
│  ┌───────────────────────▼─────────────────────────────────┼───────────────────────────┐   │
│  │                   SYSTEM 2: EAGLE BACKBONE (VLM)        │                           │   │
│  ├─────────────────────────────────────────────────────────┼───────────────────────────┤   │
│  │                                                         │                           │   │
│  │  ┌─────────────────────────────────────────┐            │                           │   │
│  │  │         SigLIP-2 Vision Encoder         │            │                           │   │
│  │  │  - Processes images at native aspect    │            │                           │   │
│  │  │  - Outputs: Image Tokens (B, N_img, D)  │            │                           │   │
│  │  └────────────────────┬────────────────────┘            │                           │   │
│  │                       │                                 │                           │   │
│  │                       ▼                                 │                           │   │
│  │  ┌─────────────────────────────────────────┐            │                           │   │
│  │  │         Vision-Language Projector       │            │                           │   │
│  │  │            (MLP1 Projection)            │            │                           │   │
│  │  │  Projects vision features → LLM space   │            │                           │   │
│  │  └────────────────────┬────────────────────┘            │                           │   │
│  │                       │                                 │                           │   │
│  │                       ▼                                 │                           │   │
│  │  ┌─────────────────────────────────────────┐            │                           │   │
│  │  │      Qwen-2 Language Model (2B)         │            │                           │   │
│  │  │  - 16 layers (configurable)             │            │                           │   │
│  │  │  - Flash Attention 2                    │            │                           │   │
│  │  │  - Top 4 layers tunable in finetuning   │            │                           │   │
│  │  │  - Hidden dim: 2048                     │            │                           │   │
│  │  │                                         │            │                           │   │
│  │  │  Input: [Text Tokens] + [Image Tokens]  │            │                           │   │
│  │  │  Output: backbone_features (B,S,2048)   │            │                           │   │
│  │  │          + image_mask, attention_mask   │            │                           │   │
│  │  └────────────────────┬────────────────────┘            │                           │   │
│  │                       │                                 │                           │   │
│  └───────────────────────┼─────────────────────────────────┼───────────────────────────┘   │
│                          │                                 │                               │
│  ┌───────────────────────▼─────────────────────────────────▼───────────────────────────┐   │
│  │                  SYSTEM 1: DIFFUSION TRANSFORMER (Action Head)                      │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                        FEATURE ENCODING STAGE                                 │  │   │
│  │  ├───────────────────────────────────────────────────────────────────────────────┤  │   │
│  │  │                                                                               │  │   │
│  │  │   backbone_features ─────► VL LayerNorm ─────► vl_embeds (B, S, 2048)         │  │   │
│  │  │                                                                               │  │   │
│  │  │   normalized_state  ─────► CategorySpecificMLP ─────► state_features          │  │   │
│  │  │   (B, T_s, D_state)       (embodiment-conditioned)    (B, T_s, 1536)          │  │   │
│  │  │                                                                               │  │   │
│  │  │                           ┌─────────────────────────────────────┐              │  │   │
│  │  │   CategorySpecificMLP:    │  W[embodiment_id] @ input + b       │              │  │   │
│  │  │   Per-embodiment weights  │  Separate params for each robot    │              │  │   │
│  │  │   max_num_embodiments=32  │  (supports multi-robot training)   │              │  │   │
│  │  │                           └─────────────────────────────────────┘              │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                    FLOW MATCHING DIFFUSION PROCESS                            │  │   │
│  │  ├───────────────────────────────────────────────────────────────────────────────┤  │   │
│  │  │                                                                               │  │   │
│  │  │   TRAINING: Predict velocity from noisy actions                               │  │   │
│  │  │   INFERENCE: Denoise random noise to actions in 4 steps                       │  │   │
│  │  │                                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                   AlternateVLDiT ARCHITECTURE (32 Layers)                     │  │   │
│  │  ├───────────────────────────────────────────────────────────────────────────────┤  │   │
│  │  │                                                                               │  │   │
│  │  │   Input: sa_embs (state + action tokens)                                      │  │   │
│  │  │   Cross-attn context: vl_embeds (vision-language features)                    │  │   │
│  │  │                                                                               │  │   │
│  │  │   Block 0 (Cross-Attn):  Attend to TEXT tokens                                │  │   │
│  │  │   Block 1 (Self-Attn):   Self-attention over state+action                     │  │   │
│  │  │   Block 2 (Cross-Attn):  Attend to IMAGE tokens                               │  │   │
│  │  │   Block 3 (Self-Attn):   Self-attention over state+action                     │  │   │
│  │  │   ... (continues alternating pattern for 32 layers)                           │  │   │
│  │  │                                                                               │  │   │
│  │  │   Output: Final hidden states → action_decoder → predicted actions            │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          OUTPUT ACTION DECODING                                      │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                      │   │
│  │   predicted_normalized_action ─────► StateActionProcessor.unapply_action            │   │
│  │   (B, T_a, D_action)                 - Unnormalize from [-1,1]                       │   │
│  │                      │               - Convert relative → absolute (if configured)  │   │
│  │                      ▼                                                               │   │
│  │                 Raw Actions                                                          │   │
│  │   (joint positions, EEF poses, gripper states, etc.)                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components Breakdown

### System 2: Eagle VLM Backbone

The Eagle backbone (`gr00t/model/modules/eagle_backbone.py`) is responsible for:
- Processing multi-view camera images through SigLIP-2 vision encoder
- Tokenizing and embedding language instructions
- Fusing vision and language through a Qwen-2 language model

**Key Features:**
- Native aspect ratio image processing (no padding/distortion)
- Flash Attention 2 for efficient computation
- Only top 4 LLM layers are fine-tuned during post-training

### System 1: Diffusion Transformer (DiT)

The action head (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`) uses flow matching to generate actions:
- **32 transformer layers** (doubled from N1.5's 16 layers)
- **AlternateVLDiT**: Alternates cross-attention between text and image tokens
- **Multi-embodiment support**: Per-robot projection layers

---

## Tokenization & Embeddings

### Image Tokenization (Eagle VLM)

```
Input: RGB Images (B, T_vid, H, W, 3) uint8
  ↓
1. Per-frame processing through SigLIP-2:
   - Flexible resolution (native aspect ratio, no padding)
   - Patch embedding: 14x14 patches → 1152-dim features
  ↓
2. Vision-Language Projector (MLP1):
   - Projects 1152 → 2048 (LLM hidden dim)
  ↓
Output: Image Tokens (B, N_img_tokens, 2048)
(N_img_tokens = T_vid × (H/14 × W/14))
```

### Text Tokenization (Qwen-2)

```
Input: Language instruction "Pick up the cup"
  ↓
1. Chat template formatting (conversation format)
2. BPE tokenization (Qwen tokenizer)
3. Token embedding lookup (vocab_size → 2048)
  ↓
Output: Text Tokens (B, N_text_tokens, 2048)
```

### State Encoding (CategorySpecificMLP)

```
Input: Normalized state (B, T_state, D_state) float32
  ↓
1. Min-max normalize to [-1, 1] (or sin/cos encode for joint angles)
2. CategorySpecificMLP (per-embodiment weights):
   layer1: D_state → 1024 (with ReLU)
   layer2: 1024 → 1536
  ↓
Output: State Features (B, T_state, 1536)
```

### Action Encoding (MultiEmbodimentActionEncoder)

```
Input: Noisy action trajectory (B, T_action, D_action) + timestep (B,)
  ↓
1. Action projection: W1[emb_id] @ action → (B, T_action, 1536)
2. Sinusoidal timestep encoding → (B, T_action, 1536)
3. Concatenate: [action_emb; time_emb] → (B, T_action, 3072)
4. MLP: W3(swish(W2(concat))) → (B, T_action, 1536)
  ↓
Output: Action Features (B, T_action, 1536)
```

---

## Model Configuration

From `gr00t/configs/model/gr00t_n1d6.py`:

```python
@dataclass
class Gr00tN1d6Config:
    # Backbone (Eagle VLM)
    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    backbone_embedding_dim: int = 2048
    select_layer: int = 16  # Use first 16 layers of Qwen
    tune_top_llm_layers: int = 4  # Finetune top 4 LLM layers
    
    # Action Head dimensions
    max_state_dim: int = 29
    max_action_dim: int = 29
    action_horizon: int = 16  # Predict 16 future timesteps
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    
    # Diffusion Transformer (DiT) - 32 layers (vs 16 in N1.5)
    diffusion_model_cfg: dict = {
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 48,  # 32 × 48 = 1536 hidden dim
        "interleave_self_attention": True,
    }
    
    # Flow matching parameters
    num_inference_timesteps: int = 4  # Only 4 denoising steps!
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    
    # Multi-embodiment support
    max_num_embodiments: int = 32
```

---

## Training Pipeline

### 1. Data Loading (ShardedSingleStepDataset)

```
LeRobot Format Dataset
├── videos/
│   └── video.{episode}_{camera}.mp4
├── data/
│   └── chunk-{idx}/episode_{idx}.parquet (states, actions, language)
└── meta/
    └── stats.json (normalization statistics)

Sharding Strategy:
- Split episodes into timesteps
- Subsample with episode_sampling_rate (default: 10%)
- Balance across shards for uniform batch sizes
```

### 2. Data Processing (Gr00tN1d6Processor)

For each timestep:

**a) Video Processing:**
- Load frames according to delta_indices (e.g., [-1, 0] for 2 frames)
- Apply augmentations: ColorJitter, RandomCrop, Resize
- Stack multiple camera views

**b) State Processing (StateActionProcessor.apply_state):**
- Load state values for each joint group
- Normalize using dataset statistics (min/max or mean/std)
- Optional: sin/cos encoding for joint angles
- Pad to max_state_dim (29)

**c) Action Processing (StateActionProcessor.apply_action):**
- Load future action chunk (e.g., 16 timesteps)
- Convert absolute → relative actions (if configured)
- Normalize to [-1, 1]
- Pad to max_action_dim (29) and max_action_horizon (40)
- Create action_mask for valid dimensions/timesteps

**d) Language Processing:**
- Lowercase and remove punctuation (formalize_language=True)
- Apply chat template

### 3. Forward Pass (Training)

```python
# Inside action_head.forward():
1. vl_embeds = vlln(backbone_features)  # LayerNorm
2. state_features = state_encoder(state, embodiment_id)
3. noise = randn_like(actions)
4. t = sample_from_beta(1.5, 1.0) * 0.999
5. noisy_trajectory = (1-t)*noise + t*actions
6. velocity_target = actions - noise
7. action_features = action_encoder(noisy_trajectory, t, emb_id)
8. sa_embs = cat([state_features, action_features], dim=1)
9. model_output = dit(sa_embs, vl_embeds, t)
10. pred_velocity = action_decoder(model_output, emb_id)
11. loss = mse(pred_velocity, velocity_target) * action_mask
```

### 4. Optimization

- **Optimizer**: AdamW
- **Learning rate**: 2e-5 (default)
- **Warmup**: 3% of training
- **Distributed**: DeepSpeed ZeRO-2
- **Gradient accumulation** for large effective batch sizes

---

## Inference Pipeline

### 1. Observation Input

```python
observation = {
    "video": {"camera_name": ndarray(B, T, H, W, 3) uint8},
    "state": {"joint_group": ndarray(B, T, D) float32},
    "language": {"task": [["pick up the cup"]] * B},
}
```

### 2. Preprocessing

```python
processor.eval()  # No augmentations during inference
# - Transform images (resize, normalize)
# - Normalize states
# - Tokenize language
# - Collate into batch
```

### 3. Backbone Forward

```python
with torch.inference_mode():
    backbone_outputs = model.backbone(inputs)

# Returns:
# - backbone_features: (B, seq_len, 2048)
# - backbone_attention_mask: (B, seq_len)
# - image_mask: (B, seq_len) - True for image tokens
```

### 4. Action Generation (Flow Matching Denoising)

```python
# Initialize with pure noise
actions = torch.randn(B, action_horizon, action_dim)
dt = 1.0 / num_inference_timesteps  # 1/4 = 0.25

# Encode state features (done once)
state_features = state_encoder(state, embodiment_id)
vl_embeds = vlln(backbone_features)

# 4-step denoising loop
for t in [0.0, 0.25, 0.5, 0.75]:
    t_discrete = int(t * 1000)  # [0, 250, 500, 750]
    
    # Encode current noisy actions
    action_features = action_encoder(actions, t_discrete, emb_id)
    sa_embs = cat([state_features, action_features], dim=1)
    
    # Cross-attend to vision-language context
    model_output = dit(sa_embs, vl_embeds, t_discrete)
    
    # Decode predicted velocity
    pred_velocity = action_decoder(model_output, emb_id)
    
    # Euler integration step
    actions = actions + dt * pred_velocity

# After 4 steps: actions is the denoised prediction
```

### 5. Action Decoding

```python
processor.decode_action(normalized_action, embodiment_tag, state)

# Steps:
# 1. Split concatenated action into joint groups
# 2. Unnormalize from [-1, 1] to physical units
# 3. Convert relative → absolute actions (if configured)

# Returns: dict[str, ndarray(B, T_action, D)] in physical units
# e.g., {"left_arm": radians, "right_arm": radians, "gripper": [0,1]}
```

### 6. Action Chunking Execution

```python
# Execute first N actions from chunk
for t in range(execution_horizon):
    env.step(action[:, t, :])

# Then get new observation and repeat
```

---

## Key Implementation Files

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| **Main Model** | `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` | `Gr00tN1d6`, `Gr00tN1d6ActionHead` |
| **Eagle Backbone** | `gr00t/model/modules/eagle_backbone.py` | `EagleBackbone` |
| **DiT (Diffusion Transformer)** | `gr00t/model/modules/dit.py` | `DiT`, `AlternateVLDiT`, `BasicTransformerBlock` |
| **Embodiment-Conditioned MLP** | `gr00t/model/modules/embodiment_conditioned_mlp.py` | `CategorySpecificMLP`, `MultiEmbodimentActionEncoder` |
| **Data Processor** | `gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py` | `Gr00tN1d6Processor`, `Gr00tN1d6DataCollator` |
| **State/Action Normalization** | `gr00t/data/state_action/state_action_processor.py` | `StateActionProcessor` |
| **Policy Interface** | `gr00t/policy/gr00t_policy.py` | `Gr00tPolicy`, `Gr00tSimPolicyWrapper` |
| **Training** | `gr00t/experiment/trainer.py` | `Gr00tTrainer` |
| **Dataset** | `gr00t/data/dataset/sharded_single_step_dataset.py` | `ShardedSingleStepDataset` |
| **Config** | `gr00t/configs/model/gr00t_n1d6.py` | `Gr00tN1d6Config` |

---

## Key Architectural Decisions in N1.6 vs N1.5

| Feature | N1.5 | N1.6 |
|---------|------|------|
| VLM Backbone | Llama-based | Eagle (Cosmos-Reason-2B variant) |
| DiT Layers | 16 | **32** |
| Post-VLM Adapter | 4-layer transformer | **Removed** (unfreezes top 4 VLM layers instead) |
| Action Representation | Absolute positions | **State-relative deltas** |
| Image Processing | Fixed resolution | **Native aspect ratio** |
| Denoising Steps | 4 | 4 |
| Parameters | ~3B | ~3B |

---

## Flow Matching vs Standard Diffusion

GR00T uses **Flow Matching** (also called Rectified Flow), which is simpler than standard DDPM:

### Standard Diffusion (DDPM)
```
Forward: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
Reverse: Predict ε, complex variance schedules
```

### Flow Matching (GR00T)
```
Forward: x_t = (1-t) * ε + t * x_0  (simple linear interpolation!)
Reverse: Predict velocity v = x_0 - ε
Update: x_{t+dt} = x_t + dt * v  (Euler integration)
```

### Benefits of Flow Matching
- Simpler training objective (MSE on velocity)
- Faster inference (only 4 steps needed)
- More stable training
- Straight-line trajectories in latent space

---

## Summary

The key innovations in GR00T N1.6 are:

1. **Eagle VLM** with native aspect ratio image processing
2. **AlternateVLDiT** that alternates attention between text and image tokens
3. **Flow matching** for fast 4-step action generation
4. **Multi-embodiment support** with per-robot projection layers
5. **State-relative actions** for better generalization across robots and tasks

The architecture effectively combines the reasoning capabilities of large vision-language models (System 2) with the precise continuous control capabilities of diffusion models (System 1), enabling robots to understand complex instructions and execute precise manipulation tasks.
