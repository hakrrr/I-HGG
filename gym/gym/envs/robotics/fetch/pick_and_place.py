import os
import random
from torchvision.utils import save_image
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np

from vae.import_vae import goal_set_fetch_pick_0
from vae.import_vae import goal_set_fetch_pick_1

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')

# Change to normal hgg
# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit robot_env: get_obs
# edit here: sample_goal


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    '''
    def _viewer_setup(self):
    body_id = self.sim.model.body_name2id('robot0:gripper_link')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        self.viewer.cam.lookat[idx] = value
    self.viewer.cam.distance = 1.
    self.viewer.cam.azimuth = 180.
    self.viewer.cam.elevation = 90.
    '''

    def _sample_goal_new(self):
        index = np.random.randint(20)
        goal_0 = goal_set_fetch_pick_0[index]
        goal_1 = goal_set_fetch_pick_1[index]
        goal_0 = self.fetch_pick_vae_0.format(goal_0)
        goal_1 = self.fetch_pick_vae_1.format(goal_1)
        save_image(goal_0.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal_0.png')
        save_image(goal_1.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal_1.png')

        x_0, y_0 = self.fetch_pick_vae_0.encode(goal_0)
        x_1, y_1 = self.fetch_pick_vae_1.encode(goal_1)
        goal_0 = self.fetch_pick_vae_0.reparameterize(x_0, y_0)
        goal_1 = self.fetch_pick_vae_1.reparameterize(x_1, y_1)
        goal_0 = goal_0.detach().cpu().numpy()
        goal_1 = goal_1.detach().cpu().numpy()

        goal = np.concatenate((np.squeeze(goal_0), np.squeeze(goal_1)))
        goal /= 9.1

        return goal.copy()

    def _get_image(self):
        rgb_array_0 = np.array(self.render(mode='rgb_array', width=84, height=84, cam_name="cam_0"))
        rgb_array_1 = np.array(self.render(mode='rgb_array', width=84, height=84, cam_name="cam_1"))
        tensor_0 = self.fetch_pick_vae_0.format(rgb_array_0)
        tensor_1 = self.fetch_pick_vae_1.format(rgb_array_1)
        x_0, y_0 = self.fetch_pick_vae_0.encode(tensor_0)
        x_1, y_1 = self.fetch_pick_vae_1.encode(tensor_1)
        obs_0 = self.fetch_pick_vae_0.reparameterize(x_0, y_0)
        obs_1 = self.fetch_pick_vae_1.reparameterize(x_1, y_1)
        obs_0 = obs_0.detach().cpu().numpy()
        obs_1 = obs_1.detach().cpu().numpy()

        obs = np.concatenate((np.squeeze(obs_0), np.squeeze(obs_1)))
        obs /= 9.1

        save_image(tensor_0.cpu().view(-1, 3, 84, 84), 'fetch_pick_0.png')
        save_image(tensor_1.cpu().view(-1, 3, 84, 84), 'fetch_pick_1.png')
        return obs

    def _generate_state(self):
        if self.visible:
            self._set_arm_visible(False)
            self.visible = False
        goal = [random.uniform(1.15, 1.45), random.uniform(0.6, 1.0), random.uniform(0.43, .7)]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal[:3]
        object_qpos[3:] = [1, 0, 0, 0]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        for _ in range(5):
            self.sim.step()
        # Check if inside checkbox:
        pos = self.sim.data.get_joint_qpos('object0:joint')
        if pos[0] < 1.15 or pos[0] > 1.45 or pos[1] < 0.6 or pos[1] > 1.0 or pos[2] < 0.43 or pos[2] > .7:
            self._generate_state()

        self._step_callback()
