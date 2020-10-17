import sys

import numpy as np
from PIL import Image

from envs import make_env
from algorithm.replay_buffer import Trajectory

class NormalLearner:
	def __init__(self, args):
		self.count = 0
		self.train_data = np.empty([1280 * 9, 84, 84, 3])
		pass

	def learn(self, args, env, env_test, agent, buffer, write_goals=0):
		for _ in range(args.episodes):
			obs = env.reset()
			current = Trajectory(obs)
			for timestep in range(args.timesteps):
				action = agent.step(obs, explore=True)
				obs, reward, done, _ = env.step(action)
				if timestep==args.timesteps-1: done = True
				current.store_step(action, obs, reward, done)
				if done: break
				# self.generateTrainData(timestep, _)
				'''
								if timestep % 10 == 0 and self.count < (1280 * 9):
					rgb_array = np.array(env.render(mode='rgb_array', width=84, height=84, cam_name='cam_0'))
					# Image.fromarray(rgb_array).show()
					self.train_data[self.count] = rgb_array
					self.count += 1
					if self.count % 1000 == 0:
						print('Count: ', self.count)

				if self.count == 1280 * 9:
					np.random.shuffle(self.train_data)
					np.save('data/Hand_Env/vae_train_data_pen', self.train_data)
					print('Finished!')
					self.count += 1
					sys.exit()
				'''

			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				agent.target_update()
		# TODO: deleted this duplicate test section
		""" 
		for _ in range(args.test_rollouts):
			def test_rollout(env, prefix=''):
				rewards = 0.0
				obs = env.reset()
				for timestep in range(args.timesteps):
					action, info = agent.step(obs, explore=False, test_info=True)
					args.logger.add_dict(info, prefix)
					obs, reward, done, info = env.step(action)
					rewards += reward
					if timestep==args.timesteps-1: done = True
					if done: break
				args.logger.add_dict(info, prefix)

			if args.goal_based:
				# goal-based envs test
				test_rollout(env, 'train/')
				test_rollout(env_test, 'test/')
			else:
				# normal envs test
				test_rollout(env)
		"""