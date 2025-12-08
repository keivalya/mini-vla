# mini-VLA Architecture Diagrams

This document provides detailed architecture diagrams for the four main components of mini-VLA. These diagrams are essential for understanding the model structure and planning future scale-up efforts.

## Table of Contents
1. [Vision Encoder](#1-vision-encoder-architecture)
2. [Text Encoder](#2-text-encoder-architecture)
3. [Fusion Module](#3-fusion-module-architecture)
4. [Diffusion Head](#4-diffusion-head-architecture)
5. [Complete Pipeline](#5-complete-vla-pipeline)

---

## 1. Vision Encoder Architecture

The Vision Encoder (ImageEncoderTinyCNN) processes camera images and extracts visual features.

### High-Level Flow

```
Input Image (B, 3, H, W)
         |
         v
   ┌─────────────┐
   │   Conv2D    │  3 → 32 channels, kernel=5, stride=2
   │   + ReLU    │
   └─────────────┘
         |
         v
   ┌─────────────┐
   │   Conv2D    │  32 → 64 channels, kernel=3, stride=2
   │   + ReLU    │
   └─────────────┘
         |
         v
   ┌─────────────┐
   │   Conv2D    │  64 → 128 channels, kernel=3, stride=2
   │   + ReLU    │
   └─────────────┘
         |
         v
   ┌─────────────┐
   │  Global     │  Average pooling over spatial dimensions
   │  Avg Pool   │
   └─────────────┘
         |
         v
   ┌─────────────┐
   │   Linear    │  128 → d_model (default: 128)
   └─────────────┘
         |
         v
   ┌─────────────┐
   │ LayerNorm   │
   └─────────────┘
         |
         v
   Output: (B, d_model)
```

### Detailed Layer Specifications

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | (B, 3, H, W) | (B, 32, H/2, W/2) | kernel=5, stride=2, padding=2 |
| Conv2 | (B, 32, H/2, W/2) | (B, 64, H/4, W/4) | kernel=3, stride=2, padding=1 |
| Conv3 | (B, 64, H/4, W/4) | (B, 128, H/8, W/8) | kernel=3, stride=2, padding=1 |
| GAP | (B, 128, H/8, W/8) | (B, 128) | Global average pooling |
| Linear | (B, 128) | (B, 128) | Projection layer |
| LayerNorm | (B, 128) | (B, 128) | Normalization |

### Key Design Choices

- **Progressive downsampling**: Each conv layer reduces spatial dimensions by 2x
- **Channel expansion**: 3 → 32 → 64 → 128 channels
- **Global Average Pooling**: Reduces spatial dimensions to 1x1, creating a compact representation
- **LayerNorm**: Stabilizes the output distribution

### Example Dimensions

For a 64x64 input image:
- Input: (B, 3, 64, 64)
- After Conv1: (B, 32, 32, 32)
- After Conv2: (B, 64, 16, 16)
- After Conv3: (B, 128, 8, 8)
- After GAP: (B, 128)
- Output: (B, 128)

---

## 2. Text Encoder Architecture

The Text Encoder (TextEncoderTinyGRU) processes tokenized text instructions.

### High-Level Flow

```
Input Tokens (B, T_text)
         |
         v
   ┌─────────────┐
   │  Embedding  │  vocab_size → d_word (default: 64)
   └─────────────┘
         |
         v
   (B, T_text, d_word)
         |
         v
   ┌─────────────┐
   │     GRU     │  Recurrent processing
   │   Layer     │  d_word → d_model (default: 128)
   └─────────────┘
         |
         v
   Extract last hidden state
         |
         v
   (B, d_model)
         |
         v
   ┌─────────────┐
   │ LayerNorm   │
   └─────────────┘
         |
         v
   Output: (B, d_model)
```

### Detailed Component Breakdown

```
┌──────────────────────────────────────────┐
│         Text Encoder Pipeline             │
│                                           │
│  Token IDs                                │
│    [10, 45, 23, 67, ...]                 │
│            │                              │
│            v                              │
│  ┌──────────────────┐                    │
│  │  Lookup Table    │  Each ID → vector  │
│  │  [vocab_size,    │                    │
│  │   d_word]        │                    │
│  └──────────────────┘                    │
│            │                              │
│            v                              │
│  Word Embeddings                          │
│  (B, T_text, 64)                         │
│            │                              │
│            v                              │
│  ┌──────────────────────────────────┐   │
│  │         GRU Cell (t=0)           │   │
│  │  ┌─────────┐    ┌─────────┐    │   │
│  │  │ Reset   │    │ Update  │    │   │
│  │  │ Gate    │    │ Gate    │    │   │
│  │  └─────────┘    └─────────┘    │   │
│  │         │            │          │   │
│  │         └────────────┘          │   │
│  │               │                 │   │
│  │         ┌─────────┐            │   │
│  │         │ Hidden  │            │   │
│  │         │ State   │            │   │
│  │         └─────────┘            │   │
│  └──────────────────────────────────┘   │
│            │                              │
│            v                              │
│  ... (repeat for each timestep)          │
│            │                              │
│            v                              │
│  Final Hidden State (B, 128)             │
│            │                              │
│            v                              │
│  LayerNorm                                │
│            │                              │
│            v                              │
│  Output (B, 128)                          │
└──────────────────────────────────────────┘
```

### GRU Internal Mechanism

```
At each timestep t:

Input: x_t (current word), h_{t-1} (previous hidden state)

┌────────────────────────────────────────┐
│  Reset Gate:                            │
│  r_t = σ(W_r @ [h_{t-1}, x_t])        │
└────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────┐
│  Update Gate:                           │
│  z_t = σ(W_z @ [h_{t-1}, x_t])        │
└────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────┐
│  Candidate Hidden State:                │
│  h̃_t = tanh(W @ [r_t * h_{t-1}, x_t]) │
└────────────────────────────────────────┘
                 │
                 v
┌────────────────────────────────────────┐
│  New Hidden State:                      │
│  h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t│
└────────────────────────────────────────┘
```

### Key Design Choices

- **GRU over LSTM**: Simpler, fewer parameters, faster training
- **Single layer**: Keeps model lightweight
- **Last hidden state**: Captures the complete instruction meaning
- **batch_first=True**: Input shape is (batch, sequence, features)

---

## 3. Fusion Module Architecture

The Fusion Module (FusionMLP) combines the three modality embeddings into a unified representation.

### High-Level Flow

```
Input: img_token (B, 128), txt_token (B, 128), state_token (B, 128)
                        |                |                |
                        └────────────────┴────────────────┘
                                         |
                                         v
                                   Concatenate
                                         |
                                         v
                                 (B, 3 * d_model)
                                   (B, 384)
                                         |
                                         v
                              ┌──────────────────┐
                              │   Linear Layer   │  384 → 128
                              └──────────────────┘
                                         |
                                         v
                              ┌──────────────────┐
                              │      ReLU        │
                              └──────────────────┘
                                         |
                                         v
                              ┌──────────────────┐
                              │   Linear Layer   │  128 → 128
                              └──────────────────┘
                                         |
                                         v
                              ┌──────────────────┐
                              │   LayerNorm      │
                              └──────────────────┘
                                         |
                                         v
                                Output: (B, d_model)
                                      (B, 128)
                                 Fused Context Vector
```

### Detailed View

```
┌─────────────────────────────────────────────────────┐
│                  Fusion Module                       │
│                                                      │
│   Vision Token          Text Token      State Token │
│   ┌─────────┐          ┌─────────┐     ┌─────────┐│
│   │  128D   │          │  128D   │     │  128D   ││
│   └────┬────┘          └────┬────┘     └────┬────┘│
│        │                     │                │     │
│        │  "what robot sees"  │ "what to do"   │     │
│        │                     │                │     │
│        └──────────┬──────────┴────────────────┘    │
│                   │                                  │
│                   v                                  │
│        ┌──────────────────────┐                    │
│        │   Concatenate        │                    │
│        │   [img, txt, state]  │                    │
│        └──────────────────────┘                    │
│                   │                                  │
│              (B, 384)                               │
│                   │                                  │
│                   v                                  │
│        ┌──────────────────────┐                    │
│        │  FC: 384 → 128       │                    │
│        │  W₁ ∈ R^{128×384}    │                    │
│        │  b₁ ∈ R^{128}        │                    │
│        └──────────────────────┘                    │
│                   │                                  │
│                   v                                  │
│        ┌──────────────────────┐                    │
│        │  ReLU Activation     │                    │
│        │  f(x) = max(0, x)    │                    │
│        └──────────────────────┘                    │
│                   │                                  │
│                   v                                  │
│        ┌──────────────────────┐                    │
│        │  FC: 128 → 128       │                    │
│        │  W₂ ∈ R^{128×128}    │                    │
│        │  b₂ ∈ R^{128}        │                    │
│        └──────────────────────┘                    │
│                   │                                  │
│                   v                                  │
│        ┌──────────────────────┐                    │
│        │  LayerNorm           │                    │
│        │  normalize across    │                    │
│        │  feature dimension   │                    │
│        └──────────────────────┘                    │
│                   │                                  │
│                   v                                  │
│           (B, 128)                                  │
│      Fused Context                                  │
│                                                      │
│  "Combined understanding of:                        │
│   - What the robot sees                             │
│   - What it needs to do                             │
│   - Where it currently is"                          │
└─────────────────────────────────────────────────────┘
```

### Why This Design?

**Simple but Effective:**
- Concatenation preserves all information from each modality
- Two-layer MLP learns cross-modal interactions
- ReLU introduces non-linearity for complex patterns
- LayerNorm stabilizes the output

**Future Improvements (for scaling):**
- Could use cross-attention instead of concatenation
- Could add residual connections
- Could use multi-head attention for better fusion
- Current design: Simple and interpretable

---

## 4. Diffusion Head Architecture

The Diffusion Head generates robot actions through an iterative denoising process.

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Diffusion Policy Head                           │
│                                                              │
│  Training Phase:                                             │
│                                                              │
│  Ground Truth      Random          Random                   │
│  Action x₀    +    Noise ε    →    Noisy Action x_t        │
│  (B, 4)           (B, 4)           (B, 4)                   │
│                                         │                     │
│                                         v                     │
│                        ┌────────────────────────────┐       │
│                        │   Denoise Model            │       │
│                        │   ε_θ(x_t, t, cond)       │       │
│                        └────────────────────────────┘       │
│                                         │                     │
│                                         v                     │
│                                  Predicted Noise ε̂          │
│                                         │                     │
│                                         v                     │
│                              Loss = MSE(ε, ε̂)               │
│                                                              │
│──────────────────────────────────────────────────────────────│
│                                                              │
│  Inference Phase (Sampling):                                 │
│                                                              │
│  Random Noise                                                │
│  x_T ~ N(0, I)                                              │
│       │                                                      │
│       v                                                      │
│  ┌─────────┐                                                │
│  │  t = T  │ ───► Denoise ───► x_{T-1}                     │
│  └─────────┘                       │                         │
│       ...                           v                         │
│  ┌─────────┐                  ┌─────────┐                  │
│  │  t = 1  │ ───► Denoise ───►│  x_0    │                  │
│  └─────────┘                  └─────────┘                  │
│                                     │                         │
│                                     v                         │
│                            Clean Action (B, 4)               │
└─────────────────────────────────────────────────────────────┘
```

### Denoise Model Architecture

```
Input: x_t (B, action_dim), t (B,), cond (B, cond_dim)
       (noisy action)        (timestep)  (fused context)
              │                   │              │
              │                   v              │
              │         ┌─────────────────┐     │
              │         │  Sinusoidal     │     │
              │         │  Time Embedding │     │
              │         │  (timestep → 32D)│     │
              │         └─────────────────┘     │
              │                   │              │
              │                   v              │
              └───────────┬───────┴──────────────┘
                          │
                          v
                  Concatenate All
                  [x_t, t_emb, cond]
                          │
                          v
                  (B, action_dim + 32 + cond_dim)
                  (B, 4 + 32 + 128) = (B, 164)
                          │
                          v
               ┌─────────────────┐
               │  Linear: 164→128 │
               │  + ReLU          │
               └─────────────────┘
                          │
                          v
               ┌─────────────────┐
               │  Linear: 128→128 │
               │  + ReLU          │
               └─────────────────┘
                          │
                          v
               ┌─────────────────┐
               │  Linear: 128→4   │
               │  (action_dim)    │
               └─────────────────┘
                          │
                          v
              Predicted Noise ε̂_θ
                    (B, 4)
```

### Sinusoidal Time Embedding

```
Purpose: Encode which diffusion step we're at (t ∈ [0, T-1])

Input: t (B,) - integer timesteps

Process:
┌──────────────────────────────────────────────┐
│  Generate frequencies:                        │
│  f = exp(linspace(log(1), log(1000), 16))   │
│                                               │
│  For each frequency:                          │
│  args = t * f                                │
│                                               │
│  Compute sin and cos:                         │
│  emb = [sin(args), cos(args)]                │
│                                               │
│  Output: (B, 32) dimensional embedding       │
└──────────────────────────────────────────────┘

Why this works:
- Different frequencies capture time at different scales
- Sinusoidal functions are periodic and smooth
- Model can learn which timestep it's processing
```

### Diffusion Forward Process (Training)

```
Given: Clean action x₀, timestep t

Step 1: Get noise schedule values
        α_t = 1 - β_t
        ᾱ_t = ∏(α_i) from i=0 to t

Step 2: Sample noise
        ε ~ N(0, I)

Step 3: Create noisy action
        x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε

Example with numbers:
  Clean action: [0.5, -0.3, 0.8, -0.2]
  ᾱ_t = 0.7 at step t
  Noise: [0.1, -0.4, 0.2, 0.3]

  x_t = √0.7 · [0.5,-0.3,0.8,-0.2] + √0.3 · [0.1,-0.4,0.2,0.3]
      ≈ [0.47, -0.47, 0.78, 0.0]
```

### Diffusion Reverse Process (Sampling)

```
Start: x_T ~ N(0, I)  (pure random noise)

For t = T-1, T-2, ..., 1, 0:

    1. Predict noise: ε̂ = ε_θ(x_t, t, cond)

    2. Predict clean action:
       x̂₀ = (x_t - √(1-ᾱ_t) · ε̂) / √(ᾱ_t)

    3. Add noise back (except at t=0):
       if t > 0:
           x_{t-1} = √(α_t) · x̂₀ + √(β_t) · ε_random
       else:
           x_0 = x̂₀  (final clean action)

Example trajectory:
  x_16: [0.8, -0.5, 0.3, 0.9]     (random noise)
  x_15: [0.6, -0.4, 0.25, 0.7]    (less noisy)
  x_14: [0.5, -0.35, 0.2, 0.6]    (less noisy)
  ...
  x_1:  [0.48, -0.32, 0.82, -0.18] (almost clean)
  x_0:  [0.50, -0.30, 0.80, -0.20] (clean action!)
```

### Beta Schedule

```
Controls how much noise to add at each step

┌────────────────────────────────────────────┐
│  Linear Schedule (used in mini-VLA):       │
│                                             │
│  β_start = 1e-4                            │
│  β_end = 1e-2                              │
│  T = 16 steps                              │
│                                             │
│  β_t = β_start + (β_end - β_start)·(t/T)  │
│                                             │
│  Noise level                               │
│  ^                                          │
│  │     ┌─────────────                      │
│  │    /                                    │
│  │   /                                     │
│  │  /                                      │
│  │ /                                       │
│  └──────────────────────> time step t     │
│   0   4   8   12  16                       │
│                                             │
│  More noise added as t increases           │
└────────────────────────────────────────────┘
```

### Key Design Decisions

1. **T=16 steps**: Balance between quality and speed
2. **Linear beta schedule**: Simple and effective
3. **DDPM sampling**: Standard denoising diffusion
4. **MSE loss**: Directly optimizes noise prediction

---

## 5. Complete VLA Pipeline

### End-to-End Data Flow

```
┌───────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                            │
│                                                                │
│   Camera Image         Text Instruction       Robot State     │
│   (B, 3, 84, 84)      "push object"          (B, 39)         │
│         │                    │                    │            │
└─────────┼────────────────────┼────────────────────┼───────────┘
          │                    │                    │
          v                    v                    v
┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
│ Vision        │   │ Text           │   │ State            │
│ Encoder       │   │ Encoder        │   │ Encoder          │
│ (TinyCNN)     │   │ (TinyGRU)      │   │ (MLP)            │
│               │   │                │   │                  │
│ Conv→Conv→    │   │ Embed→GRU→     │   │ Linear→ReLU→     │
│ Conv→GAP→     │   │ Last Hidden    │   │ Linear           │
│ Linear        │   │                │   │                  │
└───────────────┘   └────────────────┘   └──────────────────┘
          │                    │                    │
          v                    v                    v
    (B, 128)              (B, 128)              (B, 128)
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               v
                  ┌─────────────────────┐
                  │   Fusion Module     │
                  │   (FusionMLP)       │
                  │                     │
                  │ Concat→Linear→ReLU→ │
                  │ Linear→LayerNorm    │
                  └─────────────────────┘
                               │
                               v
                      Fused Context
                          (B, 128)
                               │
                               v
                  ┌─────────────────────┐
                  │  Diffusion Head     │
                  │                     │
                  │  If Training:       │
                  │   - Add noise to    │
                  │     ground truth    │
                  │   - Predict noise   │
                  │   - Compute loss    │
                  │                     │
                  │  If Inference:      │
                  │   - Start from      │
                  │     random noise    │
                  │   - Denoise T steps │
                  │   - Output action   │
                  └─────────────────────┘
                               │
                               v
┌───────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                            │
│                                                                │
│                     Robot Action                               │
│                  (B, action_dim=4)                            │
│         [gripper_x, gripper_y, gripper_z, gripper_state]     │
└───────────────────────────────────────────────────────────────┘
```

### Training Loop

```
For each batch:
    1. Sample data: (image, text, state, action)

    2. Encode observations:
       img_emb = vision_encoder(image)
       txt_emb = text_encoder(text)
       state_emb = state_encoder(state)

    3. Fuse modalities:
       context = fusion(img_emb, txt_emb, state_emb)

    4. Diffusion loss:
       t = random_timestep(0, T)
       noise = random_normal()
       noisy_action = add_noise(action, t, noise)
       pred_noise = denoise_model(noisy_action, t, context)
       loss = MSE(noise, pred_noise)

    5. Backprop and update:
       loss.backward()
       optimizer.step()
```

### Inference Loop

```
Given: image, text, state

1. Encode inputs:
   context = encode_obs(image, text, state)

2. Sample action via diffusion:
   x_t = random_normal(size=action_dim)

   for t in reverse(range(T)):
       pred_noise = denoise_model(x_t, t, context)
       x_t = denoise_step(x_t, pred_noise, t)

   action = x_0  # final denoised action

3. Execute action in environment

4. Observe new state and repeat
```

### Parameter Count Breakdown

| Component | Parameters | Details |
|-----------|-----------|---------|
| Vision Encoder | ~100K | 3 conv layers + projection |
| Text Encoder | ~100K | Embedding (vocab × 64) + GRU |
| State Encoder | ~8K | 2-layer MLP (39 → 64 → 128) |
| Fusion Module | ~50K | 2-layer MLP (384 → 128 → 128) |
| Diffusion Head | ~35K | Time embedding + 3-layer MLP |
| **Total** | **~293K** | Extremely lightweight! |

### Memory Usage (Batch Size = 64)

| Stage | Memory | Note |
|-------|--------|------|
| Input images | ~2 MB | (64, 3, 84, 84) |
| Embeddings | ~128 KB | 3 × (64, 128) |
| Activations | ~500 KB | Intermediate tensors |
| Model params | ~1.2 MB | All weights |
| **Total** | **~4 MB** | Can run on CPU! |

---

## Scaling Considerations

When planning to scale up mini-VLA, consider these architectural improvements:

### 1. Vision Encoder
- **Current**: TinyCNN with 3 layers
- **Scale-up Options**:
  - Pre-trained CLIP ViT-B/16 (~86M params)
  - Pre-trained ResNet-50 (~25M params)
  - SigLIP encoder
  - Trade-off: Performance vs. speed

### 2. Text Encoder
- **Current**: GRU with embeddings
- **Scale-up Options**:
  - Pre-trained CLIP text encoder
  - BERT-base or DistilBERT
  - T5-small for instruction following
  - Trade-off: Understanding vs. efficiency

### 3. Fusion Module
- **Current**: Simple concatenation + MLP
- **Scale-up Options**:
  - Cross-attention (Transformer-style)
  - FiLM conditioning
  - Perceiver architecture
  - Trade-off: Expressiveness vs. simplicity

### 4. Diffusion Head
- **Current**: 16 steps, linear schedule
- **Scale-up Options**:
  - DDIM for fewer steps
  - Classifier-free guidance
  - Conditional flow matching
  - Trade-off: Quality vs. sampling speed

### Multi-Task Scaling

```
Current: Single task (push-v3)
         ↓
Scale: MT10 (10 tasks)
         ↓
Scale: MT50 (50 tasks)

Architectural changes needed:
- Task-conditioned embeddings
- Shared backbone + task-specific heads
- Multi-task training strategies
```

---

## Conclusion

mini-VLA's architecture is intentionally simple and modular, making it:
- **Easy to understand** - Clear data flow
- **Easy to modify** - Swappable components
- **Easy to debug** - Minimal complexity
- **Easy to extend** - Well-defined interfaces

These diagrams provide the foundation for understanding how to scale the model while maintaining its core design philosophy of simplicity and hackability.

For implementation details, see the code in:
- [`models/encoders.py`](models/encoders.py)
- [`models/fusion.py`](models/fusion.py)
- [`models/diffusion_head.py`](models/diffusion_head.py)
- [`models/vla_diffusion_policy.py`](models/vla_diffusion_policy.py)

---

*Created by: Akhil*
*Last Updated: December 2025*
