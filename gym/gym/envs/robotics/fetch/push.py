import os
import random

from gym import utils
from gym.envs.robotics import fetch_env
from vae.import_vae import goal_set_fetch_push
from vae.import_vae import vae_fetch_push
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal
# edit here: dist_threshold (0.05 original)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.63, 0.4, 1., 0., 0., 0.],  # origin 0.53
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        goal = goal_set_fetch_push[np.random.randint(7)]
        goal = vae_fetch_push.format(goal)
        save_image(goal.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal.png')
        x, y = vae_fetch_push.encode(goal)
        goal = vae_fetch_push.reparameterize(x, y)
        goal = goal.detach().cpu().numpy()
        goal = np.squeeze(goal)
        return goal.copy()

    def _get_image(self):
        np.array(self.render(mode='rgb_array',
                             width=84, height=84))
        rgb_array = np.array(self.render(mode='rgb_array',
                                         width=84, height=84))
        tensor = vae_fetch_push.format(rgb_array)
        x, y = vae_fetch_push.encode(tensor)
        obs = vae_fetch_push.reparameterize(x, y)
        obs = obs.detach().cpu().numpy()
        obs = np.squeeze(obs)
        save_image(tensor.cpu().view(-1, 3, 84, 84), 'ach_fetch_push.png')
        return obs

    def _generate_state(self):
        if self.visible:
            self._set_arm_visible(False)
            self.visible = False
        goal = [random.uniform(1.15, 1.45), random.uniform(0.6, 1.0), 0.43]
        # goal = [1.3, .7, .432]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal[:3]
        object_qpos[3:] = [1, 0, 0, 0]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        for _ in range(15):
            self.sim.step()

        # Check if inside checkbox:
        pos = self.sim.data.get_joint_qpos('object0:joint').copy()
        if pos[0] < 1.15 or pos[0] > 1.45 or pos[1] < 0.6 or pos[1] > 1.0 or pos[2] < 0.42 or pos[2] > .7:
            self._generate_state()
        # Image.fromarray(np.array(self.render(mode='rgb_array', width=300, height=300, cam_name="cam_0"))).show()

        # latent = self._get_image()
