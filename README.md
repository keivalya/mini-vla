# Mini-VLA



## Collect data

```
python -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name corner \
  --episodes 50 \
  --max-steps 150 \
  --output-path data/metaworld_push_bc.npz
  ```

## Train

```
python -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 128 \
  --save-path checkpoints/model.pt \
  --device cpu
```

## Test

```
python -m scripts.test \
  --checkpoint checkpoints/model.pt \
  --env-name push-v3 \
  --episodes 5 \
  --max-steps 150 \
  --instruction "push the object to the goal" \
  --device cpu \
  --save-video \
  --video-dir videos
```

## Inference

(to be documented)