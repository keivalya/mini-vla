"""Test VLA Diffusion Policy on Meta-World MT1"""

import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer
from models.vision.registry import VisionEncoderCfg


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
        help="Random seed for the environment",
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
    ckpt = torch.load(checkpoint_path, map_location=device)

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

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
        vision_cfg=vision_cfg,
        use_flow_matching=use_flow_matching,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    return model, tokenizer


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model + tokenizer
    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T_text)

    # environment
    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.policy_camera_name,
    )
    video_camera_name = args.video_camera_name or args.policy_camera_name
    video_rotation = resolve_video_rotation(
        policy_camera_name=args.policy_camera_name,
        video_camera_name=video_camera_name,
        requested_rotation=args.video_rotate,
    )
    video_env = None
    if args.save_video and video_camera_name != args.policy_camera_name:
        video_env = MetaWorldMT1Wrapper(
            env_name=args.env_name,
            seed=args.seed,
            render_mode="rgb_array",
            camera_name=video_camera_name,
        )

    print(f"[test] Meta-World MT1 env: {args.env_name}")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")
    print(f"[test] policy_camera={args.policy_camera_name}, video_camera={video_camera_name}")
    print(f"[test] video_rotate={video_rotation}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # evaluation
    for ep in range(args.episodes):
        img, state, info = env.reset()
        step = 0
        ep_reward = 0.0

        if video_env is not None:
            try:
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
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W)
            state_t = torch.from_numpy(state).float().unsqueeze(0) # (1, state_dim)

            img_t = img_t.to(device)
            state_t = state_t.to(device)

            # inference
            with torch.no_grad():
                action_t = model.act(img_t, text_ids, state_t)  # (1, action_dim)
            action_np = action_t.squeeze(0).cpu().numpy()

            # step environment
            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            step += 1

            if video_env is not None:
                video_env.sync_from(env)
                frames.append(rotate_frame(video_env.render().copy(), video_rotation))
            else:
                frames.append(rotate_frame(img.copy(), video_rotation))

        print(f"[test] Episode {ep+1}/{args.episodes}: reward={ep_reward:.3f}, steps={step}")

        # save video
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"{args.env_name}_ep{ep+1:03d}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {video_path}")

    env.close()
    if video_env is not None:
        video_env.close()
    print("[test] Done.")


if __name__ == "__main__":
    main()
