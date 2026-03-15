# Action Normalization And Clipping

This note explains the action normalization upgrade added to mini-VLA, why it matters, what changed in code, and what to run now.

## Why This Change Was Needed

Before this change, the model was trained directly on raw environment actions.

That is usually a poor default because:

- different action dimensions can have different scales
- optimization becomes unnecessarily harder
- generated actions can drift outside the simulator's legal action range

The new behavior is:

- train on normalized actions
- save action normalization statistics in the checkpoint
- unnormalize predicted actions at inference time
- clip final actions to the environment action bounds before stepping the env

## What Changed

### Training

In [scripts/train.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/train.py):

- the dataset now computes per-dimension `action_mean`
- the dataset now computes per-dimension `action_std`
- actions are normalized before being passed to the model
- the checkpoint now stores:
  - `action_stats.mean`
  - `action_stats.std`
  - `action_stats.eps`

Formula used during training:

```text
normalized_action = (raw_action - action_mean) / action_std
```

`action_std` is clamped with a small epsilon so division stays numerically stable.

### Inference

In [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py):

- checkpoints now load `action_stats` if present
- model outputs are converted back to raw environment actions
- actions are clipped to the env action space before `env.step(...)`

Formula used during inference:

```text
raw_action = predicted_normalized_action * action_std + action_mean
clipped_action = clip(raw_action, action_low, action_high)
```

### Environment Wrapper

In [envs/metaworld_env.py](/Users/keivalya/Desktop/Projects/mini-vla/envs/metaworld_env.py):

- the wrapper now exposes:
  - `action_low`
  - `action_high`

These come from the simulator action space and are used for clipping.

## Behavior With Old Checkpoints

Old checkpoints do not contain `action_stats`.

Current fallback behavior:

- `action_mean = 0`
- `action_std = 1`

That keeps old checkpoints loadable, but it does **not** give you the benefit of action normalization. To actually use this upgrade, retrain the model.

## Example

Assume one action dimension has:

```text
action_mean = 0.25
action_std = 0.50
```

If the model predicts:

```text
predicted_normalized_action = -1.20
```

Then inference converts it back to env space as:

```text
raw_action = (-1.20 * 0.50) + 0.25 = -0.35
```

If the environment bounds for that dimension are:

```text
action_low = -1.0
action_high = 1.0
```

Then the clipped action stays:

```text
-0.35
```

If instead the reconstructed action were `1.7`, it would be clipped to `1.0`.

## Files Changed

- [scripts/train.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/train.py)
- [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py)
- [envs/metaworld_env.py](/Users/keivalya/Desktop/Projects/mini-vla/envs/metaworld_env.py)

## What You Need To Run

Because the checkpoint format changed, retrain first.

### Train a New Checkpoint

```bash
python3 -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_model_norm.pt \
  --device cpu
```

### Test the New Checkpoint

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --policy-camera-name topview \
  --video-camera-name corner2 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos_showcase
```

## What You Should Expect In Logs

During training, you should now see action statistics printed once at startup:

```text
[train] action_mean= [...]
[train] action_std= [...]
```

During testing, you should now see:

```text
[test] action_low= [...]
[test] action_high= [...]
[test] action_mean= [...]
[test] action_std= [...]
```

That confirms the checkpoint and environment bounds are being used correctly.
