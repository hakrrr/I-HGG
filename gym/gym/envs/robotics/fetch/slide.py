import os
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env


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
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal_new(self):
        goal = goal_set_fetch_push[np.random.randint(20)]
        goal = self.fetch_push_vae.format(goal)
        #save_image(goal.cpu().view(-1, 3, self.img_size, self.img_size), 'videos/goal/goal.png')
        x, y = self.fetch_push_vae.encode(goal)
        goal = self.fetch_push_vae.reparameterize(x, y)
        goal = goal.detach().cpu().numpy()
        goal = np.squeeze(goal)
        return goal.copy()

    def _get_image_new(self, img_name='default'):
        local_vae = self.fetch_push_vae
        np.array(self.render(mode='rgb_array',
                             width=84, height=84))
        rgb_array = np.array(self.render(mode='rgb_array',
                                         width=84, height=84))
        tensor = local_vae.format(rgb_array)
        x, y = local_vae.encode(tensor)
        obs = local_vae.reparameterize(x, y)
        obs = obs.detach().cpu().numpy()
        obs = np.squeeze(obs)
        # save_image(tensor.cpu().view(-1, 3, 84, 84), img_name)
        return obs