import os

import cv2
import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
from common import get_args
from PIL import Image
import time
import tensorflow as tf


class Player:
    def __init__(self, args):
        # initialize environment
        self.args = args
        self.env = make_env(args)
        self.args.timesteps = self.env.env.env.spec.max_episode_steps
        self.env_test = make_env(args)
        self.info = []
        self.test_rollouts = 10

        # get current policy from path (restore tf session + graph)
        self.play_dir = args.play_path
        self.play_epoch = args.play_epoch
        self.meta_path = "{}saved_policy-{}.meta".format(self.play_dir, self.play_epoch)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.play_dir))
        graph = tf.get_default_graph()
        self.raw_obs_ph = graph.get_tensor_by_name("raw_obs_ph:0")
        self.pi = graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")

    def my_step_batch(self, obs):
        # compute actions from obs based on current policy by running tf session initialized before
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions

    def play(self):
        # play policy on env
        env = self.env
        acc_sum, obs = 0.0, []
        for i in range(self.test_rollouts):
            obs.append(goal_based_process(env.reset()))
            goal_img_rgb = Image.open('videos/goal/goal_0.png')

            for timestep in range(25):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                # if not timestep % 1:
                rgb_array = np.array(env.render(mode='rgb_array', width=512, height=512, cam_name="cam_0"))

                path = 'videos/frames/frame_' + str(i * 50 + timestep) + '.png'
                rgb_array = self.get_concat_h(Image.fromarray(rgb_array), goal_img)
                rgb_array.save(path)
                    # Image.fromarray(rgb_array).show()
                    # time.sleep(.3)
                    #for proc in psutil.process_iter():
                    #    if proc.name() == "display":
                    #        proc.kill()

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def make_video(self, path_to_folder, ext_end):
        image_files = [f for f in os.listdir(path_to_folder) if f.endswith(ext_end)]
        image_files.sort(key=lambda x: int(x.replace('frame_', '').replace(ext_end, '')))
        img_array = []
        for filename in image_files:
            img = cv2.imread(path_to_folder + filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter('videos/hand_reach.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == "__main__":
    # Call play.py in order to see current policy progress
    args = get_args()
    player = Player(args)
    player.play()
    player.make_video('videos/frames/', '.png')
