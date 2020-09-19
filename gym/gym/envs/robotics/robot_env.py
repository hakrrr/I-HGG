import os
import copy
import numpy as np
from PIL import Image
import gym
import sys
import torch
from gym import error, spaces
from gym.utils import seeding
from torchvision.utils import save_image

from vae.import_vae import vae_fetch_push, vae_fetch_slide
from vae.import_vae import vae_fetch_pick_0
# from vae.import_vae import vae_fetch_pick_1
from vae.import_vae import vae_fetch_reach
# from vae.import_vae import vae_fetch_slide

from vae.import_vae import vae_egg
from vae.import_vae import vae_block
from vae.import_vae import vae_pen
from vae.import_vae import vae_hand_reach

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # Vae Model assignment
        # self.fetch_push_vae = vae_fetch_push
        # self.fetch_pick_vae_0 = vae_fetch_pick_0
        # self.fetch_pick_vae_1 = vae_fetch_pick_1
        # self.fetch_slide_vae = vae_fetch_slide
        # self.fetch_reach = vae_fetch_reach

        # self.hand_vae_egg = vae_egg
        # self.hand_vae_block = vae_block
        # self.hand_vae_pen = vae_pen
        # self.hand_vae_reach = vae_hand_reach

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # Debug: Check if VAE encodes correctly
        # ach1 = torch.from_numpy(obs['achieved_goal']).float().to('cuda')
        # save_image(vae_fetch_slide.decode(ach1).view(-1, 3, 84, 84), 'ach_latent.png')
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()

        # Generate state
        '''
        size = 100
        train_data_0 = np.empty([size, 84, 84, 3])
        # train_data_1 = np.empty([1280, 84, 84, 3])
        for i in range(size):
            self._generate_state()
            img_0 = self.render(width=84, height=84, cam_name='cam_0')
            # img_1 = self.render(width=84, height=84, cam_name='cam_1')
            # img_one = Image.fromarray(img_0, 'RGB')
            # img_two = Image.fromarray(img_1, 'RGB')
            # img_one.show()
            # img_two.show()
            train_data_0[i] = img_0
            # train_data_1[i] = img_1
            if i % 1000 == 0:
                print(i)
        np.save('data/Fetch_Env/Idk.npy', train_data_0)
        # np.save('data/Fetch_Env/vae_goal_pick_1', train_data_1)
        print('Finished')
        sys.exit()
        '''
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, cam_name="cam_1"):
        # Visualize target
        # self._render_callback()
        rgb_array = self.sim.render(width=width, height=height, camera_name=cam_name)
        rgb_array = np.rot90(rgb_array)
        # img = Image.fromarray(rgb_array, 'RGB')
        # img.show()
        return rgb_array

    '''
    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, cam_name='camera_1'):
    if mode == 'rgb_array':
        x = self.sim.render(width=width, height=height, camera_name=cam_name)
        img = Image.fromarray(x, 'RGB')
        img.show()
        return self.sim.render(width=width, height=height, camera_name=cam_name)

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
    # self._render_callback()
    x = self.sim.render(width=100, height=100, camera_name="camera_1")
    if mode == 'rgb_array':
        self._get_viewer(mode).render(width, height)
        # window size used for old mujoco-py:
        data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'human':
        self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    '''

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _generate_state(self):
        pass
