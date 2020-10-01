import os
import random

from PIL import Image
from torchvision.utils import save_image

from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from vae.import_vae import goal_set_fetch_reach

# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal !
# edit here: dist_threshold (optional)
# edit robot_env: render (between hand and fetch env)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal_old(self):
        goal = goal_set_fetch_reach[np.random.randint(5)+10]
        goal = self.fetch_reach.format(goal)
        save_image(goal.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal.png')
        x, y = self.fetch_reach.encode(goal)
        goal = self.fetch_reach.reparameterize(x, y)
        goal = goal.detach().cpu().numpy()
        goal = np.squeeze(goal)
        return goal.copy()

    def _get_image(self, img_name='default'):
        local_vae = self.fetch_reach
        np.array(self.render(mode='rgb_array',
                             width=84, height=84))
        rgb_array = np.array(self.render(mode='rgb_array',
                                         width=84, height=84))
        tensor = local_vae.format(rgb_array)
        x, y = local_vae.encode(tensor)
        obs = local_vae.reparameterize(x, y)
        obs = obs.detach().cpu().numpy()
        obs = np.squeeze(obs)
        save_image(tensor.cpu().view(-1, 3, 84, 84), img_name)
        return obs

    def _generate_state(self):
        goal = [random.uniform(1.15, 1.45), random.uniform(0.6, 1.0), 0.43]
        self._set_gripper(goal)
        self.sim.forward()
        for _ in range(15):
            self.sim.step()
        self._step_callback()

        # Image.fromarray(np.array(self.render(mode='rgb_array', width=300, height=300, cam_name="cam_0"))).show()
        # latent = self._get_image()

