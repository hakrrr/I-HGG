# Exploration via Image-Based Hindsight Goal Generation

This is the TensorFlow implementation for the Bachelor's thesis "Image-Based Hindsight Goal Generation via VAE for Robotic Object Manipulation with Sparse-Reward Deep Reinforcement Learning" (James Li, 2020). 
It is based on the implementation of G-HGG (Matthias Brucker, 2020).



## Requirements
1. Ubuntu 16.04 (newer versions such as 18.04 should work as well)
2. Python 3.5.2 (newer versions such as 3.6.8 should work as well)
3. MuJoCo == 1.50 (see instructions on https://github.com/openai/mujoco-py)
4. Install requirements
5. Download the directory 'data' from [https://syncandshare.lrz.de/getlink/fiMKUN6s6mENZfDh3MQkcSR1/] and place it into the project root directory
6. Download the directory 'figure' from [https://syncandshare.lrz.de/getlink/fiMKUN6s6mENZfDh3MQkcSR1/] and place it into the project root directory for plotting the results from the paper
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
python vae/fetch_vae/vae_fetch_push.py
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
python plot.py figures/FetchPush --naming 0 --e_per_c 20
```

## Playing 

To generate a video looking at the agent solving the respective task according to his learned policy, issue the following command:

```bash
# Scheme: python play.py --env env_id --play_path log_dir --play_epoch <epoch number, latest or best>

# Example
python play_new.py --env FetchPush-v1 --play_path figures/000-ddpg-FetchPush-v1-hgg/ --play_epoch latest
```

## Vanilla HGG and HER

To train with Vanilla HGG and HER, change the function-names according to the instructions in the comments of 
each [env_name].py