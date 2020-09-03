# Exploration via Hindsight Goal Generation

This is the TensorFlow implementation for the Bachelor's thesis "Image-Based Hindsight Goal Generation via VAE for Robotic Object Manipulation with Sparse-Reward Deep Reinforcement Learning" (James Li, 2020). 
It is based on the implementation of G-HGG (Matthias Brucker, 2020).



## Requirements
1. Ubuntu 16.04 (newer versions such as 18.04 should work as well)
2. Python 3.5.2 (newer versions such as 3.6.8 should work as well)
3. MuJoCo == 1.50 (see instructions on https://github.com/openai/mujoco-py)
4. Install requirements
5. Download the directory 'data' from [link] and place it into the root directory
```bash
pip install -r requirements.txt
```

## Running commands from I-HGG

Run the following commands to reproduce results.

## Train VAE
The directory 'data' includes a set training data and trained VAE model for each task.
New VAE models can be trained with following command:
```bash
# Training a vae model for the environment
# Note: Training data are loaded in line 72. Edit the path if you wish to train with other training data 
python vae/train_fetch_pick.py
```

## Agent Training

```bash
# FetchPush
# HER (with EBP)
python train.py --tag 000 --learn normal --env=FetchPush-v1
# IGG (with EBP)
python train.py --tag 100 --learn hgg --env=FetchPush-v1


## Plotting
To plot the agent's performance on multiple training runs, copy all training run directories into one directory. For example, we put all FetchPushLabyrinth runs in a directory called BA_Labyrinth, same for FetchPickObstacle (BA_Obstacle), FetchPickNoObstacle (BA_NoObstacle) and FetchPickAndThrow (BA_Throw). naming=0 is recommended as default. For our result plot commands, have a look at create_result_figures.sh. 

```bash
# Scheme: python plot.py log_dir env_id --naming <naming_code> --e_per_c <episodes per cycle>
python plot.py figures/BA_Labyrinth FetchPushLabyrinth-v1 --naming 0 --e_per_c 20
```

## Figures

Figures and the data they are based on can be found in the directory "figures" and were generate with the following scripts:

```bash
# Result and Ablation plots (Figures are already generated in the respective subdirectories in directory "figures"):
./create_result_figures.sh

# Other plots (Figures are already generated in directory "figures"):
python create_figures.py
```

## Playing 

To look at the agent solving the respective task according to his learned policy, issue the following command:

```bash
# Scheme: python play.py --env env_id --goal custom --play_path log_dir --play_epoch <epoch number, latest or best>

# FetchPushLabyrinth
# I-HGG
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path figures/BA_Labyrinth/000-ddpg-FetchPushLabyrinth-v1-hgg-mesh-stop --play_epoch best
# HGG
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path figures/BA_Labyrinth/010-ddpg-FetchPushLabyrinth-v1-hgg-stop --play_epoch best
# HER
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path figures/BA_Labyrinth/010-ddpg-FetchPushLabyrinth-v1-normal --play_epoch best

#FetchPickObstacle
python play.py --env FetchPickObstacle-v1 --goal custom --play_path figures/BA_Obstacle/100-ddpg-FetchPickObstacle-v1-hgg-mesh-stop --play_epoch best
python play.py --env FetchPickObstacle-v1 --goal custom --play_path figures/BA_Obstacle/112-ddpg-FetchPickObstacle-v1-hgg-stop --play_epoch best
python play.py --env FetchPickObstacle-v1 --goal custom --play_path figures/BA_Obstacle/120-ddpg-FetchPickObstacle-v1-normal --play_epoch best

#FetchPickNoObstacle
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path figures/BA_NoObstacle/200-ddpg-FetchPickNoObstacle-v1-hgg-mesh-stop --play_epoch best
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path figures/BA_NoObstacle/210-ddpg-FetchPickNoObstacle-v1-hgg-stop --play_epoch best
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path figures/BA_NoObstacle/220-ddpg-FetchPickNoObstacle-v1-normal --play_epoch best

#FetchPickAndThrow
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path figures/BA_Throw/300a-ddpg-FetchPickAndThrow-v1-hgg-mesh-stop --play_epoch best
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path figures/BA_Throw/310a-ddpg-FetchPickAndThrow-v1-hgg-stop --play_epoch best
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path figures/BA_Throw/320a-ddpg-FetchPickAndThrow-v1-hgg-normal --play_epoch best
```

## Running commands from HGG paper

Run the following commands to reproduce our main results shown in section 5.1 of the HGG paper.

```bash
python train.py --tag='HGG_fetch_push' --env=FetchPush-v1
python train.py --tag='HGG_fetch_pick' --env=FetchPickAndPlace-v1
python train.py --tag='HGG_hand_block' --env=HandManipulateBlock-v0
python train.py --tag='HGG_hand_egg' --env=HandManipulateEgg-v0
```
