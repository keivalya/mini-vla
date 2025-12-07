import gymnasium as gym
import numpy as np
import metaworld

class MetaWorldMT1Wrapper:
    """
    Wraps a Metaworld MT1 environment into a simple interface:
    - reset() -> (image, state)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, env_name='push-v3', seed=42, render_mode='rgb_array', camera_name='topview'):
        self.env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            camera_name=camera_name
        )
        self.render_mode = render_mode

        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """
        Adapt this to your env's observation structure.
        Examples:
          - obs might be a dict with keys ["robot_state", "object_state"].
          - or it might already be a flat vector.
        """
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                state_parts = []
                if "robot_state" in obs:
                    state_parts.append(obs["robot_state"])
                if "object_state" in obs:
                    state_parts.append(obs["object_state"])
                state = np.concatenate(state_parts, axis=-1)
            else:
                raise KeyError(
                    f"No suitable state keys in observation dict. "
                    f"Available keys: {list(obs.keys())}"
                )
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        img = self.env.render()
        img = img.astype(np.uint8)
        return img

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()