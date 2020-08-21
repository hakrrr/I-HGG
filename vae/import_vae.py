import numpy as np
import vae.train_fetch_push_vae as VAE_PUSH
import vae.train_hand_egg_vae as VAE_EGG
import vae.train_hand_reach_vae as VAE_HAND_REACH

vae_fetch_push = VAE_PUSH.load_Vae(path='data/FetchPush/vae_model_goal')
vae_egg = VAE_EGG.load_Vae(path='data/HandManipulate/vae_model_egg')
vae_hand_reach = VAE_HAND_REACH.load_Vae(path='data/HandManipulate/vae_model_reach')

goal_set_fetch_push = np.load('data/FetchPush/goal_set.npy')
goal_set_egg = np.load('data/HandManipulate/egg_goal_set.npy')
goal_set_reach = np.load('data/HandManipulate/reach_goal_set.npy')

