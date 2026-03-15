# Stronger Action Head

This note documents the stronger action-head upgrade added to mini-VLA.

## What Changed

Previously, both policy heads used a very small MLP:

- linear
- ReLU
- linear
- ReLU
- linear

That is cheap, but it is also a bottleneck. The action head is where the model turns fused context into actual control outputs, so making it slightly stronger is a high-value change.

The new action head is a shared residual MLP used by both:

- diffusion action denoising
- flow-matching velocity prediction

## New Architecture

The new shared head uses:

- input projection
- SiLU activation
- multiple residual MLP blocks
- LayerNorm before the final projection

Concretely:

```text
input -> Linear -> SiLU -> Residual Blocks -> LayerNorm -> SiLU -> Linear -> action output
```

Each residual block is:

```text
x -> LayerNorm -> Linear -> SiLU -> Linear -> +x
```

This gives the head:

- more capacity
- better gradient flow than the old shallow MLP
- stronger conditioning for the same encoders

## Files Changed

- [models/action_head_utils.py](/Users/keivalya/Desktop/Projects/mini-vla/models/action_head_utils.py)
- [models/diffusion_head.py](/Users/keivalya/Desktop/Projects/mini-vla/models/diffusion_head.py)
- [models/flow_matching_head.py](/Users/keivalya/Desktop/Projects/mini-vla/models/flow_matching_head.py)
- [models/vla_diffusion_policy.py](/Users/keivalya/Desktop/Projects/mini-vla/models/vla_diffusion_policy.py)
- [scripts/train.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/train.py)
- [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py)

## New CLI Option

Training now supports:

```text
--action-head-hidden-dim
```

This controls the hidden width of the stronger residual action head.

Default:

```text
256
```

That value is saved in the checkpoint and automatically reused at test time.

## Important Compatibility Note

This upgrade changes the actual parameterization of the action head.

That means older checkpoints trained with the old shallow head should be treated as incompatible for this feature. Retrain the model with the new code.

## How To Train

Example with flow matching, action normalization, and 4-step observation history:

```bash
python3 -m scripts.train \
  --dataset-path data/push_v3_history.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --obs-history-len 4 \
  --action-head-hidden-dim 256 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_history4_stronghead.pt \
  --device cpu
```

If you want a larger head, increase it:

```bash
python3 -m scripts.train \
  --dataset-path data/push_v3_history.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --obs-history-len 4 \
  --action-head-hidden-dim 384 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_history4_head384.pt \
  --device cpu
```

## How To Test

You do not need to pass `--action-head-hidden-dim` at test time. The script reads it from the checkpoint automatically.

Example:

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_history4_stronghead.pt \
  --env-name push-v3 \
  --seeds 42 43 44 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --policy-camera-name topview \
  --video-camera-name corner2 \
  --device cpu \
  --save-video \
  --video-dir videos_stronghead
```

The test logs now print:

```text
[test] action_head_hidden_dim=256
```

That confirms the checkpoint is loading the stronger head configuration correctly.

## When To Use This

Use the stronger head when:

- the model underfits even with decent encoders
- actions look noisy or weakly conditioned
- you want more capacity in the control head before changing the encoders

## Minimal Design Choice

This change intentionally does not:

- introduce attention in the action head
- add a transformer decoder
- change the diffusion or flow-matching objective

It only strengthens the final action network with a small residual MLP, which is the lowest-risk capacity upgrade in this codebase.
