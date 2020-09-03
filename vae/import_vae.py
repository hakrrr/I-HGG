import numpy as np
import vae.train_fetch_push_vae as VAE_PUSH
import vae.train_fetch_push_reach as VAE_FETCH_REACH
import vae.train_fetch_push_pick as VAE_FETCH_PICK
import vae.train_fetch_push_slide as VAE_FETCH_SLIDE

import vae.train_hand_egg_vae as VAE_EGG
import vae.train_hand_reach_vae as VAE_HAND_REACH
import vae.train_hand_block_vae as VAE_HAND_BLOCK
import vae.train_hand_pen_vae as VAE_HAND_PEN

# Vae Model
vae_fetch_push = VAE_PUSH.load_Vae(path='data/FetchPush/vae_model_push')
vae_fetch_reach = VAE_FETCH_REACH.load_Vae(path='data/FetchPush/vae_model_reach')
vae_fetch_pick = VAE_FETCH_PICK.load_Vae(path='data/FetchPush/vae_model_pick')
vae_fetch_slide = VAE_FETCH_SLIDE.load_Vae(path='data/FetchPush/vae_model_slide')

vae_egg = VAE_EGG.load_Vae(path='data/HandManipulate/vae_model_egg')
vae_block = VAE_HAND_BLOCK.load_Vae(path='data/HandManipulate/vae_model_block')
vae_pen = VAE_HAND_PEN.load_Vae(path='data/HandManipulate/vae_model_pen')
vae_hand_reach = VAE_HAND_REACH.load_Vae(path='data/HandManipulate/vae_model_reach')


# Goalset
goal_set_fetch_push = np.load('data/FetchPush/push_goal_set.npy')
goal_set_fetch_reach = np.load('data/FetchPush/reach_goal_set.npy')
goal_set_fetch_pick = np.load('data/FetchPush/pick_goal_set.npy')
goal_set_fetch_slide = np.load('data/FetchPush/slide_goal_set.npy')

goal_set_egg = np.load('data/HandManipulate/egg_goal_set.npy')
goal_set_block = np.load('data/HandManipulate/block_goal_set.npy')
goal_set_pen = np.load('data/HandManipulate/pen_goal_set.npy')
goal_set_reach = np.load('data/HandManipulate/reach_goal_set_500.npy')


# Setup new goalset
#old_training_data = np.load('data/FetchPush/vae_train_data_push.npy')
#index = np.random.choice(old_training_data.shape[0], 20, replace=False)
#goal_set = old_training_data[index]
#np.save('data/FetchPush/push_goal_set.npy', goal_set)
