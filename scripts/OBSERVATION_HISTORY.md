# Observation History

This note documents the minimal observation-history upgrade added to mini-VLA.

## What Was Implemented

Instead of giving the policy only the current image and current state, the model can now consume a short history of recent observations.

This implementation uses:

- stacked image history
- stacked state history
- the same current action target

The goal is to give the policy short-term temporal context without redesigning the policy head.

## Why Observation History Was Chosen

Between the two options:

- observation history
- action chunking

observation history is the smaller change in this codebase. It keeps the action head unchanged and only modifies data preparation plus input encoding.

## What Changed

### 1. Dataset collection now saves episode boundaries

In [scripts/collect_data.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/collect_data.py):

- each sample now stores an `episode_id`

This is required so history stacking does not accidentally cross episode boundaries.

### 2. Training supports `--obs-history-len`

In [scripts/train.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/train.py):

- added `--obs-history-len`
- the dataset now builds a history window for images and states
- the beginning of each episode is padded by repeating the earliest available observation in that episode
- the checkpoint now stores `obs_history_len`

### 3. The model now encodes stacked observations

In [models/vla_diffusion_policy.py](/Users/keivalya/Desktop/Projects/mini-vla/models/vla_diffusion_policy.py):

- image history is encoded frame-by-frame with the same vision encoder
- the resulting per-frame image embeddings are concatenated and projected back to `d_model`
- state history is concatenated and passed through the state encoder

Default behavior is unchanged when `obs_history_len=1`.

### 4. Inference now matches training

In [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py):

- the script reads `obs_history_len` from the checkpoint
- it maintains rolling image and state buffers during evaluation
- the model receives the same history length at test time that it was trained with

## Important Compatibility Note

Observation history greater than `1` requires datasets that include `episode_ids`.

Older datasets may not have that field.

Current behavior:

- `obs_history_len=1` works with old datasets
- `obs_history_len>1` raises an error if `episode_ids` are missing

If that happens, regenerate the dataset with the updated [scripts/collect_data.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/collect_data.py).

## Example

If you set:

```text
obs_history_len = 4
```

then, for each training or evaluation step, the policy sees:

- the last 4 images
- the last 4 states

At the first step of an episode, there is no past history yet, so the first observation is repeated to fill the buffer.

So the first few episode steps look like:

Step 0:

```text
[obs0, obs0, obs0, obs0]
```

Step 1:

```text
[obs0, obs0, obs0, obs1]
```

Step 2:

```text
[obs0, obs0, obs1, obs2]
```

This keeps history well-defined from the first step onward.

## Files Changed

- [scripts/collect_data.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/collect_data.py)
- [scripts/train.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/train.py)
- [models/vla_diffusion_policy.py](/Users/keivalya/Desktop/Projects/mini-vla/models/vla_diffusion_policy.py)
- [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py)

## What To Run

### 1. Regenerate the dataset

This step is required if your current dataset does not contain `episode_ids`.

```bash
python3 -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name topview \
  --episodes 100 \
  --max-steps 100 \
  --instruction "push the object to the goal" \
  --output-path data/push_v3_history.npz
```

### 2. Train with observation history

Example with 4-step history:

```bash
python3 -m scripts.train \
  --dataset-path data/push_v3_history.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --obs-history-len 4 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_history4.pt \
  --device cpu
```

### 3. Test the checkpoint

No extra history flag is needed at test time. The script reads it from the checkpoint.

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_history4.pt \
  --env-name push-v3 \
  --seeds 42 43 44 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --policy-camera-name topview \
  --video-camera-name corner2 \
  --device cpu \
  --save-video \
  --video-dir videos_history4
```

## Minimal Design Choices

This implementation intentionally does not:

- change the action head
- add a recurrent model
- add attention over time
- change the environment interface

It only adds short observation history with the least amount of code movement needed to keep train and test aligned.
