import os
import numpy as np
import random

import torch
from PIL import Image
from gym import utils
from gym.envs.robotics import fetch_env
from torchvision.utils import save_image
from vae.import_vae import goal_set_fetch_slide
from vae.import_vae import vae_fetch_slide

# Change to normal hgg
# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal
# edit here: dist_threshold (original: 0.05)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.08,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        # Sample randomly from goalset
        index = np.random.randint(5)
        goal_0 = goal_set_fetch_slide[index]
        goal_0 = vae_fetch_slide.format(goal_0)
        # save_image(goal_0.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal.png')
        x_0, y_0 = vae_fetch_slide.encode(goal_0)
        goal_0 = vae_fetch_slide.reparameterize(x_0, y_0)
        goal_0 = goal_0.detach().cpu().numpy()
        goal = np.squeeze(goal_0)
        # ach1 = torch.from_numpy(goal_0).float().to('cuda')
        # save_image(vae_fetch_slide.decode(ach1).view(-1, 3, 84, 84), 'ach_latent.png')

        return goal.copy()

    def _get_image(self):
        rgb_array_0 = np.array(self.render(mode='rgb_array', width=84, height=84, cam_name="cam_0"))
        tensor_0 = vae_fetch_slide.format(rgb_array_0)
        x_0, y_0 = vae_fetch_slide.encode(tensor_0)
        obs_0 = vae_fetch_slide.reparameterize(x_0, y_0)
        obs_0 = obs_0.detach().cpu().numpy()
        obs = np.squeeze(obs_0)
        # save_image(tensor_0.cpu().view(-1, 3, 84, 84), 'fetch_slide_0.png')
        return obs

    def _generate_state(self):
        if self.visible:
            self._set_arm_visible(False)
            self.visible = False
        goal = [random.uniform(.8, 1.51), random.uniform(0.5, 1), 0.422]
        # For the Goal set
        # goal = [1.36, random.uniform(0.6, .9), .422]
        # goal = [1.36, .94, .422]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal[:3]
        object_qpos[3:] = [1, 0, 0, 0]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        for _ in range(3):
            self.sim.step()

        # Check if inside checkbox:
        pos = self.sim.data.get_joint_qpos('object0:joint').copy()
        if pos[0] < .94 or pos[0] > 1.45 or pos[1] < 0.4 or pos[1] > 2 or pos[2] < 0.4 or pos[2] > .7:
            self._generate_state()
        # Image.fromarray(np.array(self.render(mode='rgb_array', width=300, height=300, cam_name="cam_0"))).show()
        # latent = self._get_image()

        self._step_callback()
