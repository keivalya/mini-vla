<div align="center">

# mini-VLA

</div>

mini-VLA is a minimal, beginner-friendly Vision-Language-Action (VLA) model designed to show how modern robot policies can fuse images, text instructions, and robot states to produce continuous actions.

> [!IMPORTANT]
> it is far from being production-ready! Purpose? Education. Optimizations? Yes, I am developing them. Checkout other branches.

<div align="center">
  <img src="./assets/corner2_push.gif" alt="push the object to the goal">
</div>

This project intentionally keeps the codebase small (~150 LOC for the core model) so that,
- beginners can understand the complete VLA training and exec pipeline
- researchers can rapidly prototype new ideas around this
- students can learn diffusion-based action generation w/o heavy dependencies

> [!TIP]
> I recommend reading the following blogs to get started with mini-VLA implementation.
> - [Building Vision-Language-Action Model from scratch (Basics)](https://open.substack.com/pub/keivalya/p/building-vision-language-action-from?utm_campaign=post-expanded-share&utm_medium=web)
> - [Building VLA models from scratch — II (Math, Code, and Intuition)](https://medium.com/@keivalyap/building-vla-models-from-scratch-ii-0180020dbc85)

This project is not meant to be state-of-the-art instead, it provides a clear, hackable template for understanding VLA design.

The mini-VLA model core is mainly four files: [models/encoders.py](models/encoders.py) contains encoders for images, text and states corresponding to the robot, [models/fusion.py](models/fusion.py) simply combines vision-language-action embeddings using an MLP (yeah, not ideal but simple and it works OKAY), [models/diffusion_head.py](models/diffusion_head.py) generates action using diffusion policy, and [models/vla_diffusion_policy.py](models/vla_diffusion_policy.py) combines everything!

Additionally, I provide scripts such as [scripts/collect_data.py](scripts/collect_data.py) to collect data using an expert policy, [scripts/train.py](scripts/train.py) to train the VLA on the collected data, and [scripts/test.py](scripts/test.py) to test VLA-Diffusion Policy's performance (+ save videos).


## Getting started

Create (or activate) a conda environment

```
conda create --name mini-vla python=3.10
conda activate mini-vla
```

Clone mini-VLA project

```
git clone https://github.com/keivalya/mini-vla.git
cd mini-vla
```

Install dependencies

```
pip install -r requirements.txt
```

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

much more! Checkout [mini-vla/issues](https://github.com/keivalya/mini-vla/issues).