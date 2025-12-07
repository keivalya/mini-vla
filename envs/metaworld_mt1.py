# Metaworld/MT1

import gymnasium as gym
import metaworld
import time
from metaworld.policies import ENV_POLICY_MAP

seed = 42
# ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
camera_name = 'corner'
env_name = 'push-v3'
env = gym.make(
    'Meta-World/MT1',
    env_name=env_name,
    seed=seed,
    render_mode='human',
    camera_name=camera_name
)

obs, info = env.reset()
policy = ENV_POLICY_MAP[env_name]()

done = False
while not done:
    action = policy.get_action(obs)
    obs, reward, truncate, terminate, info = env.step(action)
    
    env.render()
    time.sleep(1/60)

    done = int(info['success']) == 1

    if done:
        obs, info = env.reset()
        done = False