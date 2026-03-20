"""Train VLA on dataset of image, state, action, and text instruction"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vla_diffusion_policy import VLADiffusionPolicy
from models.vision.registry import VisionEncoderCfg


ACTION_STD_EPS = 1e-6


class TrainingDataset(Dataset):
    def __init__(self, path, resize_to=64, obs_history_len=1):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to
        self.obs_history_len = obs_history_len
        self.action_mean = self.actions.mean(axis=0).astype(np.float32)
        self.action_std = self.actions.std(axis=0).astype(np.float32)
        self.action_std = np.clip(self.action_std, ACTION_STD_EPS, None)
        self.episode_ids = data["episode_ids"] if "episode_ids" in data else None
        if self.obs_history_len > 1 and self.episode_ids is None:
            raise ValueError(
                "Observation history requires `episode_ids` in the dataset. "
                "Regenerate the dataset with the updated scripts.collect_data."
            )
        if self.episode_ids is not None:
            self.episode_start_idx = {}
            for idx, episode_id in enumerate(self.episode_ids.tolist()):
                self.episode_start_idx.setdefault(int(episode_id), idx)

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return self.images.shape[0]

    def _history_indices(self, idx):
        if self.obs_history_len == 1:
            return [idx]

        episode_id = int(self.episode_ids[idx])
        episode_start = self.episode_start_idx[episode_id]
        indices = []
        for offset in range(self.obs_history_len):
            hist_idx = max(episode_start, idx - (self.obs_history_len - 1 - offset))
            indices.append(hist_idx)
        return indices

    def __getitem__(self, idx):
        history_indices = self._history_indices(idx)
        images = []
        states = []
        for hist_idx in history_indices:
            img = self.images[hist_idx]  # (H, W, 3), uint8
            if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
                img = self.cv2.resize(img, (self.resize_to, self.resize_to))
            images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            states.append(torch.from_numpy(self.states[hist_idx]).float())

        img = torch.stack(images, dim=0) if self.obs_history_len > 1 else images[0]
        state = torch.cat(states, dim=0)
        normalized_action = (self.actions[idx] - self.action_mean) / self.action_std
        action = torch.from_numpy(normalized_action).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        return img, state, action, text_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--obs-history-len", type=int, default=1)
    parser.add_argument("--action-head-hidden-dim", type=int, default=256)
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")

    # vision command-line arguments
    parser.add_argument("--vision-name", type=str, default="tinycnn",
                    help="tinycnn | hf_clip_vit | hf_siglip_vit")
    parser.add_argument("--vision-pretrained", type=str, default=None,
                    help="HF model id override (optional)")
    parser.add_argument("--vision-trainable", action="store_true",
                    help="If set, HF backbone is trainable")
    parser.add_argument("--vision-image-size", type=int, default=None,
                    help="Override vision encoder image size (e.g. 224)")
    parser.add_argument("--use-flow-matching", action="store_true", default=False,
                    help="If set, use flow-matching instead of diffusion")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TrainingDataset(args.dataset_path, resize_to=args.resize_to, obs_history_len=args.obs_history_len)
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    print("[train] action_mean=", dataset.action_mean.tolist())
    print("[train] action_std=", dataset.action_std.tolist())

    # vision config takes care of loading the correct vision encoder from CLI
    vision_cfg = VisionEncoderCfg(
        name=args.vision_name,
        d_model=args.d_model,
        pretrained=args.vision_pretrained,
        trainable=args.vision_trainable,
        image_size=args.vision_image_size,
    )

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T,
        vision_cfg=vision_cfg,
        use_flow_matching=args.use_flow_matching, # flow-matching / diffusion
        obs_history_len=args.obs_history_len,
        action_head_hidden_dim=args.action_head_hidden_dim,
    ).to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for img, state, action, text_ids in loader:
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            loss = model.loss(img, text_ids, state, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
            "use_flow_matching": args.use_flow_matching,
            "obs_history_len": args.obs_history_len,
            "action_head_hidden_dim": args.action_head_hidden_dim,
            "action_stats": {
                "mean": dataset.action_mean,
                "std": dataset.action_std,
                "eps": ACTION_STD_EPS,
            },
            # save vision encoder config
            "vision_cfg": {
                "name": vision_cfg.name,
                "d_model": vision_cfg.d_model,
                "pretrained": vision_cfg.pretrained,
                "trainable": vision_cfg.trainable,
                "image_size": vision_cfg.image_size,
            },
        },
        args.save_path,
    )
    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()
