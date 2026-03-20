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
        self.camera_name = camera_name

        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
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

    def _get_image(self, camera_name=None):
        if camera_name is None or camera_name == self.camera_name:
            img = self.env.render()
        else:
            try:
                img = self.env.render(camera_name=camera_name)
            except TypeError:
                # Older wrappers may only support the camera selected at env creation.
                img = self.env.render()
        img = img.astype(np.uint8)
        return img

    def render(self, camera_name=None):
        return self._get_image(camera_name=camera_name)

    def _get_stateful_env(self):
        for candidate in (self.env, getattr(self.env, "unwrapped", None)):
            if candidate is not None:
                has_get = hasattr(candidate, "get_env_state")
                has_set = hasattr(candidate, "set_env_state")
                if has_get and has_set:
                    return candidate
        return None

    def get_env_state(self):
        stateful_env = self._get_stateful_env()
        if stateful_env is None:
            raise AttributeError("Environment does not expose get_env_state/set_env_state")
        return stateful_env.get_env_state()

    def set_env_state(self, state):
        stateful_env = self._get_stateful_env()
        if stateful_env is None:
            raise AttributeError("Environment does not expose get_env_state/set_env_state")
        stateful_env.set_env_state(state)

    def sync_from(self, other):
        self.set_env_state(other.get_env_state())

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
