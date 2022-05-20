# Safe-Reinforcement-Learning-for-Black-Box-Systems-Using-Reachability-Analysis
This repository contains the code for the paper [Safe Reinforcement Learning Using Black-Box Reachability Analysis](https://arxiv.org/abs/2204.07417). [vido](https://youtube.com/playlist?list=PL7bkcpwNaUjz-S1b5KBzpgCZ1SP4DXsoi)

## Installation
The repository provides two ros packages: brsl and brsl_msgs. To get everything up and running, we need to install ros and some other dependencies. Everything is tested on ubuntu 20.04 LTS

### Ros Installation.
you can find the installation steps for ros in the official [link](http://wiki.ros.org/noetic/Installation/Ubuntu).

### Libraries Installation
You need to install the following libraries.

```
python -m pip install absl-py
python -m pip install gin
python -m pip install tensorflow
python -m pip install gin-config
python -m pip install tf-agnets
python -m pip install scipy
python -m pip install joblib
python -m pip install tensorboardX
python -m pip install torch
python -m pip install omegaconf==2.0
python -m pip install hydra-core==1.1.0
python -m pip install mbrl==v0.1.4
```

If you want to utilize gpu for your training, you should install tf-gpu and torch cuda instead.


### Usage
To run the agents, first, make a catkin_ws.
```
cd
mkdir -p catkin_ws/src
cd catkin_ws
catkin_make
```

Then copy both `brsl` and `brsl_msgs` to `catkin_ws/src` in your home directory.

Finally, run:
```
cd ~/catkin_ws/src
catkin_make
```

This should build the repository.

To run the scripts, go to your catkin_ws directory, source the workspace, and run both the environment and the launch script for the agent.

```
roscore
```
Then, in a second terminal run the environment:

```
cd ~/catkin_ws
source devel/setup.bash
rosrun brsl Turtlebot_environment
```

Finally, open a third terminal and run the agent

```
cd ~/catkin_ws
source devel/setup.bash
roslaunch brsl turtlebot.launch
```
