python plot.py figures/BA_Labyrinth FetchPushLabyrinth --naming 1 --save_path figures/results_labyrinth.pdf
python plot.py figures/BA_Obstacle FetchPickObstacle --naming 1 --save_path figures/results_obstacle.pdf
python plot.py figures/BA_NoObstacle FetchPickNoObstacle --naming 1 --save_path figures/results_noobstacle.pdf
python plot.py figures/BA_Throw FetchPickAndThrow --naming 1 --save_path figures/results_throw.pdf

python plot.py figures/AP_Throw_stop FetchPickAndThrow --naming 2 --save_path figures/ablation_throw_stop.pdf
python plot.py figures/AP_Labyrinth_stop FetchPushLabyrinth --naming 3 --save_path figures/ablation_labyrinth_stop.pdf

python plot.py figures/AP_Labyrinth_n FetchPushLabyrinth --naming 4 --save_path figures/ablation_labyrinth_n.pdf
python plot.py figures/AP_Obstacle_n FetchPickObstacle --naming 5 --save_path figures/ablation_obstacle_n.pdf

python timing.py figures/AP_Throw_time FetchPickAndThrow --naming 1 --save_path figures/time.pdf