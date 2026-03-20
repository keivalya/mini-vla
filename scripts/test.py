"""Test VLA Diffusion Policy on Meta-World MT1"""

import os
import argparse
from collections import deque
import numpy as np
import torch
import imageio.v2 as imageio

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer
from models.vision.registry import VisionEncoderCfg


def normalize_action_stats(action_stats: dict | None, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if action_stats is None:
        return np.zeros(action_dim, dtype=np.float32), np.ones(action_dim, dtype=np.float32)

    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    std = np.clip(std, float(action_stats.get("eps", 1e-6)), None)
    return mean, std


def summarize_metric(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float32)
    return f"mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}"


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return frame
    if degrees % 90 != 0:
        raise ValueError(f"Rotation must be a multiple of 90 degrees, got {degrees}")
    k = (degrees // 90) % 4
    return np.rot90(frame, k=k).copy()


def resolve_video_rotation(policy_camera_name: str, video_camera_name: str, requested_rotation: int | None) -> int:
    if requested_rotation is not None:
        return requested_rotation
    # Showcase camera captures in this stack are typically upside down relative
    # to the policy view, while the policy camera should stay untouched.
    if video_camera_name != policy_camera_name:
        return 180
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/vla_diffusion_metaworld_push.pt",
        help="Path to trained VLA diffusion checkpoint",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="push-v3",
        help="Meta-World MT1 task name, e.g. push-v3, reach-v3, pick-place-v3",
    )
    parser.add_argument(
        "--policy-camera-name",
        type=str,
        default="topview",
        help="Camera used for policy observations at inference time",
    )
    parser.add_argument(
        "--video-camera-name",
        type=str,
        default=None,
        help="Optional camera used only for saved videos; defaults to the policy camera",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for single-seed evaluation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds for multi-seed evaluation; overrides --seed",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="push the object to the goal",
        help="Language instruction passed to the VLA",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="If set, save each episode as an MP4 video",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory to save videos (if --save-video is set)",
    )
    parser.add_argument(
        "--video-rotate",
        type=int,
        default=None,
        help="Optional override to rotate saved video frames by 0, 90, 180, or 270 degrees",
    )

    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]

    # load vision encoder config
    vision_cfg = None
    if "vision_cfg" in ckpt:
        vision_cfg = VisionEncoderCfg(**ckpt["vision_cfg"])

    use_flow_matching = ckpt.get("use_flow_matching", False)
    obs_history_len = ckpt.get("obs_history_len", 1)
    action_head_hidden_dim = ckpt.get("action_head_hidden_dim", 128)

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
        vision_cfg=vision_cfg,
        use_flow_matching=use_flow_matching,
        obs_history_len=obs_history_len,
        action_head_hidden_dim=action_head_hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    action_mean, action_std = normalize_action_stats(ckpt.get("action_stats"), action_dim)

    return model, tokenizer, action_mean, action_std, obs_history_len, action_head_hidden_dim


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model + tokenizer
    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, tokenizer, action_mean, action_std, obs_history_len, action_head_hidden_dim = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T_text)

    video_camera_name = args.video_camera_name or args.policy_camera_name
    video_rotation = resolve_video_rotation(
        policy_camera_name=args.policy_camera_name,
        video_camera_name=video_camera_name,
        requested_rotation=args.video_rotate,
    )
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    eval_seeds = args.seeds if args.seeds is not None else [args.seed]
    episode_rewards = []
    episode_steps = []
    episode_successes = []

    for seed in eval_seeds:
        env = MetaWorldMT1Wrapper(
            env_name=args.env_name,
            seed=seed,
            render_mode="rgb_array",
            camera_name=args.policy_camera_name,
        )
        video_env = None
        if args.save_video and video_camera_name != args.policy_camera_name:
            video_env = MetaWorldMT1Wrapper(
                env_name=args.env_name,
                seed=seed,
                render_mode="rgb_array",
                camera_name=video_camera_name,
            )

        print(f"[test] Meta-World MT1 env: {args.env_name}")
        print(f"[test] seed={seed}, state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")
        print(f"[test] obs_history_len={obs_history_len}")
        print(f"[test] action_head_hidden_dim={action_head_hidden_dim}")
        print(f"[test] policy_camera={args.policy_camera_name}, video_camera={video_camera_name}")
        print(f"[test] video_rotate={video_rotation}")
        print(f"[test] action_low={env.action_low.tolist()}")
        print(f"[test] action_high={env.action_high.tolist()}")
        print(f"[test] action_mean={action_mean.tolist()}")
        print(f"[test] action_std={action_std.tolist()}")

        for ep in range(args.episodes):
            img, state, info = env.reset(seed=seed + ep)
            step = 0
            ep_reward = 0.0
            ep_success = 0
            img_history = deque([img.copy() for _ in range(obs_history_len)], maxlen=obs_history_len)
            state_history = deque([state.copy() for _ in range(obs_history_len)], maxlen=obs_history_len)

            if video_env is not None:
                try:
                    video_env.reset(seed=seed + ep)
                    video_env.sync_from(env)
                    frames = [rotate_frame(video_env.render().copy(), video_rotation)]
                except AttributeError:
                    print("[test] Warning: env state sync is unavailable; falling back to policy camera for video.")
                    video_env.close()
                    video_env = None
                    frames = [rotate_frame(img.copy(), video_rotation)]
            else:
                frames = [rotate_frame(img.copy(), video_rotation)]

            done = False
            while not done and step < args.max_steps:
                if obs_history_len > 1:
                    img_np = np.stack(list(img_history), axis=0)  # (H, H_img, W_img, 3)
                    img_t = torch.from_numpy(img_np).permute(0, 3, 1, 2).float().unsqueeze(0) / 255.0
                else:
                    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                state_t = torch.from_numpy(np.concatenate(list(state_history), axis=0)).float().unsqueeze(0)

                img_t = img_t.to(device)
                state_t = state_t.to(device)

                with torch.no_grad():
                    action_t = model.act(img_t, text_ids, state_t)
                action_np = action_t.squeeze(0).cpu().numpy()
                action_np = action_np * action_std + action_mean
                action_np = np.clip(action_np, env.action_low, env.action_high)

                img, state, reward, done, info = env.step(action_np)
                ep_reward += reward
                ep_success = max(ep_success, int(info.get("success", 0)))
                step += 1
                img_history.append(img.copy())
                state_history.append(state.copy())

                if video_env is not None:
                    video_env.sync_from(env)
                    frames.append(rotate_frame(video_env.render().copy(), video_rotation))
                else:
                    frames.append(rotate_frame(img.copy(), video_rotation))

            episode_rewards.append(ep_reward)
            episode_steps.append(step)
            episode_successes.append(ep_success)
            print(
                f"[test] Seed {seed} Episode {ep+1}/{args.episodes}: "
                f"reward={ep_reward:.3f}, steps={step}, success={ep_success}"
            )

            if args.save_video:
                video_path = os.path.join(args.video_dir, f"{args.env_name}_seed{seed}_ep{ep+1:03d}.mp4")
                with imageio.get_writer(video_path, fps=20) as writer:
                    for f in frames:
                        writer.append_data(f)
                print(f"[test] Saved video to {video_path}")

        env.close()
        if video_env is not None:
            video_env.close()

    print(f"[test] Aggregated rewards: {summarize_metric(episode_rewards)}")
    print(f"[test] Aggregated steps: {summarize_metric(episode_steps)}")
    print(f"[test] Success rate: {100.0 * float(np.mean(episode_successes)):.1f}%")
    print(f"[test] Total episodes: {len(episode_rewards)} across seeds={eval_seeds}")
    print("[test] Done.")


if __name__ == "__main__":
    main()
