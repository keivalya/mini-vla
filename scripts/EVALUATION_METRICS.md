# Aggregate Evaluation Metrics And Multi-Seed Runs

This note documents the minimal evaluation upgrade added to [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py).

## What Changed

The evaluation script now supports:

- aggregate metrics across all evaluation episodes
- multi-seed test runs in a single command
- per-episode success tracking
- seed-specific video filenames so outputs do not overwrite each other

This was implemented with minimal scope in [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py). No new script was added.

## New CLI Behavior

### Existing single-seed behavior still works

You can still run:

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --seed 42 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu
```

### New multi-seed behavior

You can now pass multiple seeds with `--seeds`.

Example:

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --seeds 42 43 44 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu
```

If `--seeds` is provided, it overrides `--seed`.

## Metrics Now Reported

For each episode, the script now prints:

- episode reward
- episode length in steps
- episode success

At the end of the run, it prints aggregate statistics across all episodes from all seeds:

- aggregated reward: mean, std, min, max
- aggregated steps: mean, std, min, max
- overall success rate
- total number of evaluated episodes
- list of seeds used

## Success Definition

Episode success is tracked from the environment `info` dict:

```text
success = max(success, int(info.get("success", 0)))
```

That means an episode is counted as successful if the environment reports success at any step during the rollout.

## Video Naming Change

When `--save-video` is used, output videos now include the seed in the filename.

Before:

```text
push-v3_ep001.mp4
```

Now:

```text
push-v3_seed42_ep001.mp4
```

This prevents collisions during multi-seed runs.

## Example Commands

### Single-seed evaluation with saved videos

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --seed 42 \
  --policy-camera-name topview \
  --video-camera-name corner2 \
  --episodes 3 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos_eval
```

### Multi-seed evaluation without videos

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --seeds 42 43 44 45 46 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu
```

### Multi-seed evaluation with videos

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_norm.pt \
  --env-name push-v3 \
  --seeds 42 43 44 \
  --policy-camera-name topview \
  --video-camera-name corner2 \
  --episodes 2 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos_multi_seed
```

## Example Output

Per-episode logs:

```text
[test] Seed 42 Episode 1/5: reward=7.321, steps=84, success=1
[test] Seed 42 Episode 2/5: reward=6.908, steps=91, success=1
```

Aggregate summary:

```text
[test] Aggregated rewards: mean=6.944, std=0.412, min=6.201, max=7.321
[test] Aggregated steps: mean=88.400, std=5.238, min=80.000, max=97.000
[test] Success rate: 86.7%
[test] Total episodes: 15 across seeds=[42, 43, 44]
```

## Files Changed

- [scripts/test.py](/Users/keivalya/Desktop/Projects/mini-vla/scripts/test.py)

## Why This Is Useful

A single rollout can be misleading. Multi-seed evaluation gives you a more defensible estimate of how stable the learned policy actually is, and aggregate metrics make it easier to compare checkpoints without inspecting videos one by one.
