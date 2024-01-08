# autowheelchair

Using 
Ububtu 20.04
ROS Noetic 
Flatland (https://github.com/avidbots/flatland)

After all is installed put the project folder in catkin workspace src folder and run in one terminal:
roscore

In another:
roslaunch autowheelchairs_flatland world.launch
or
roslaunch autowheelchairs_flatland world_5x_speed.launch
or
roslaunch autowheelchairs_flatland world_10x_speed.launch

And in another:
rosrun autowheelchairs_flatland ma.py

Change MADDPG hyperparameters to optimize training

The globals.py file has additional instructions to change options in the program.

github link:
https://github.com/FilipeAlmeidaFEUP/autowheelchairs_flatland