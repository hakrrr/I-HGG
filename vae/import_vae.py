import numpy as np
import vae.fetch_vae.vae_fetch_push as VAE_PUSH
import vae.fetch_vae.vae_fetch_reach as VAE_FETCH_REACH
import vae.fetch_vae.vae_fetch_slide as VAE_FETCH_SLIDE
import vae.fetch_vae.vae_fetch_pick_0 as VAE_FETCH_PICK_0
import vae.fetch_vae.vae_fetch_pick_1 as VAE_FETCH_PICK_1

import vae.hand_vae.train_hand_egg_vae as VAE_EGG
import vae.hand_vae.train_hand_reach_vae as VAE_HAND_REACH
import vae.hand_vae.train_hand_block_vae as VAE_HAND_BLOCK
import vae.hand_vae.train_hand_pen_vae as VAE_HAND_PEN

# Vae Model
# Fetch
vae_fetch_push = VAE_PUSH.load_Vae(path='data/Fetch_Env/vae_model_push')
vae_fetch_reach = VAE_FETCH_REACH.load_Vae(path='data/Fetch_Env/vae_model_reach')
vae_fetch_slide = VAE_FETCH_SLIDE.load_Vae(path='data/Fetch_Env/vae_model_slide')
vae_fetch_pick_0 = VAE_FETCH_PICK_0.load_Vae(path='data/Fetch_Env/vae_model_pick_0')
# vae_fetch_pick_1 = VAE_FETCH_PICK_1.load_Vae(path='data/Fetch_Env/vae_model_pick_1')

# Hand
vae_egg = VAE_EGG.load_Vae(path='data/Hand_Env/vae_model_egg')
vae_block = VAE_HAND_BLOCK.load_Vae(path='data/Hand_Env/vae_model_block_0')
vae_pen = VAE_HAND_PEN.load_Vae(path='data/Hand_Env/vae_model_pen')
vae_hand_reach = VAE_HAND_REACH.load_Vae(path='data/Hand_Env/vae_model_reach')


# Goalset
# Fetch
goal_set_fetch_push = np.load('data/Fetch_Env/push_goal_set.npy')
goal_set_fetch_reach = np.load('data/Fetch_Env/reach_goal_set.npy')
goal_set_fetch_slide = np.load('data/Fetch_Env/slide_goal_set.npy')
goal_set_fetch_pick_0 = np.load('data/Fetch_Env/pick_goal_set_000.npy')
# goal_set_fetch_pick_1 = np.load('data/Fetch_Env/pick_goal_set_1.npy')

# Hand
goal_set_egg = np.load('data/Hand_Env/egg_goal_set.npy')
goal_set_block = np.load('data/Hand_Env/block_goal_set.npy')
goal_set_pen = np.load('data/Hand_Env/pen_goal_set.npy')
goal_set_reach = np.load('data/Hand_Env/reach_goal_set.npy')


# Setup new goalset
#old_training_data_0 = np.load('data/Hand_Env/vae_train_data_block.npy')
#old_training_data_1 = np.load('data/Fetch_Env/vae_train_data_pick_1.npy')
#index = np.random.choice(old_training_data_0.shape[0], 20, replace=False)
#goal_set_0 = old_training_data_0[index]
#goal_set_1 = old_training_data_1[index]
#np.save('data/Hand_Env/block_goal_set.npy', goal_set_0)
#np.save('data/Fetch_Env/pick_goal_set_1.npy', goal_set_1)
