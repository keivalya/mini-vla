# mini-VLA

mini-VLA is a minimal, beginner-friendly Vision-Language-Action (VLA) model designed to show how modern robot policies can fuse images, text instructions, and robot states to produce continuous actions.

This project intentionally keeps the codebase very small (≈150–200 LOC for the core model) so that,
- beginners can understand the complete pipeline
- researchers can rapidly prototype new ideas
- students can learn diffusion-based action generation without heavy dependencies

This project is not meant to be state-of-the-art — instead, it provides a clear, hackable template for understanding VLA design.

## Collect demonstration data

This gathers trajectories using an expert Meta-World policy and saves them in `.npz` dataset.

```
python -m scripts.collect_data \
  --env-name push-v3 \
  --camera-name corner \
  --episodes 100 \
  --max-steps 100 \
  --output-path data/metaworld_push_bc.npz
  ```

## Train your VLA model

Train a small vision-language diffusion policy on your collected dataset.

```
python -m scripts.train \
  --dataset-path data/push_v3.npz \
  --epochs 50 \
  --batch-size 64 \
  --save-path checkpoints/model.pt \
  --device cpu
```

## Test your model in sim

Run the trained VLA inside the Meta-World MT1 environment.

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

## Inference (coming soon)

Planning to,
- support for multiple tasks (MT10 or M50 something, let's see how much I can scale it)
- adding larger vision/text backbones (CLIP, SigLIP, ViT) -- w/o losing simplicity
- arbitrary text-input during inference

## 🙌 Contributing

PRs, improvements, and experiments are welcome! Try adding support for,
- MLP-only vision encoder
- Online evaluation metrics
- MT10 / MT50 multi-task training
much more!
