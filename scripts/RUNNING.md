# Running mini-VLA

This guide covers the exact commands to:

- collect a dataset
- train a model
- evaluate a model
- save videos from a different camera angle than the one used by the policy

Run all commands from the repository root:

```bash
cd /Users/keivalya/Desktop/Projects/mini-vla
```

Use module mode (`python3 -m ...`) so imports like `from models...` work correctly.

## 1. Dataset

An existing dataset is already present at [data/push_v3.npz](/Users/keivalya/Desktop/Projects/mini-vla/data/push_v3.npz).

If you want to regenerate it, run:

```bash
mkdir -p data

python3 -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name corner \
  --episodes 100 \
  --max-steps 100 \
  --instruction "push the object to the goal" \
  --output-path data/push_v3.npz
```

Notes:

- `--camera-name` controls the image view stored in the dataset.
- If you trained a model on top-view images, keep using top-view at inference time.

## 2. Train Flow Matching

The flow-matching bug was fixed in the code, so older flow-matching checkpoints should be considered stale and retrained.

Train a new flow-matching checkpoint with:

```bash
mkdir -p checkpoints

python3 -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_model_fixed.pt \
  --device cpu
```

If CUDA is available, replace `--device cpu` with `--device cuda`.

## 3. Evaluate Normally

This runs the policy and saves videos from the same camera used for inference.

```bash
mkdir -p videos_fm_eval

python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_fixed.pt \
  --env-name push-v3 \
  --policy-camera-name topview \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos_fm_eval
```

## 4. Evaluate With a Showcase Camera

This is the recommended setup when:

- the policy should keep using the camera it was trained on
- the saved MP4 should show the robot from a better angle for demos or social posts

Example:

```bash
mkdir -p videos_showcase

python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_fixed.pt \
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

Meaning of the camera flags:

- `--policy-camera-name`: camera used to render the image that goes into the VLA
- `--video-camera-name`: camera used only for the saved video

If `--video-camera-name` is omitted, the saved video uses the same camera as the policy.

## 5. Camera Recommendations

For model behavior:

- use the same camera view the model was trained on

For showcase videos:

- `corner`
- `corner2`
- `corner3`
- `corner4`
- `behindGripper`

Example combinations:

- trained on `topview` -> infer with `--policy-camera-name topview`
- save a nicer demo video -> add `--video-camera-name corner2`

## 6. Common Errors

### `ModuleNotFoundError: No module named 'models'`

Cause:

- running `python3 scripts/train.py` or `python3 scripts/test.py` directly

Fix:

```bash
python3 -m scripts.train ...
python3 -m scripts.test ...
```

Alternative:

```bash
PYTHONPATH=. python3 scripts/train.py ...
PYTHONPATH=. python3 scripts/test.py ...
```

### Video still looks like the policy camera

Cause:

- some Meta-World/Gym render stacks only support the camera chosen when the environment is created

Current behavior:

- the code falls back to the policy camera instead of crashing

## 7. Quick Copy-Paste Commands

Train:

```bash
python3 -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --d-model 128 \
  --diffusion-T 16 \
  --use-flow-matching \
  --save-path checkpoints/flow_matching_model_fixed.pt \
  --device cpu
```

Test with same camera:

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_fixed.pt \
  --env-name push-v3 \
  --policy-camera-name topview \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos_fm_eval
```

Test with separate showcase camera:

```bash
python3 -m scripts.test \
  --checkpoint checkpoints/flow_matching_model_fixed.pt \
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
