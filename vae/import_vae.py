import numpy as np
import vae.train_fetch_push_vae as VAE_PUSH
import vae.train_hand_egg_vae as VAE_EGG
import vae.train_hand_reach_vae as VAE_HAND_REACH
import vae.train_hand_block_vae as VAE_HAND_BLOCK
import vae.train_hand_pen_vae as VAE_HAND_PEN

vae_fetch_push = VAE_PUSH.load_Vae(path='data/FetchPush/vae_model_goal')
vae_egg = VAE_EGG.load_Vae(path='data/HandManipulate/vae_model_egg')
vae_block = VAE_HAND_BLOCK.load_Vae(path='data/HandManipulate/vae_model_block')
vae_pen = VAE_HAND_PEN.load_Vae(path='data/HandManipulate/vae_model_pen')
vae_hand_reach = VAE_HAND_REACH.load_Vae(path='data/HandManipulate/vae_model_reach')

goal_set_fetch_push = np.load('data/FetchPush/goal_set.npy')
goal_set_egg = np.load('data/HandManipulate/egg_goal_set.npy')
goal_set_block = np.load('data/HandManipulate/block_goal_set.npy')
goal_set_pen = np.load('data/HandManipulate/pen_goal_set.npy')
goal_set_reach = np.load('data/HandManipulate/reach_goal_set.npy')


# Setup new goalset
# old_training_data = np.load('data/HandManipulate/vae_train_data_pen.npy')
# index = np.random.choice(old_training_data.shape[0], 20, replace=False)
# goal_set = old_training_data[index]
# np.save('data/HandManipulate/pen_goal_set.npy', goal_set)


