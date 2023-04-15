# CarAware
A Deep Reinforcement Learning Platform for Multiple Autonomous Vehicles Based on CARLA Simulation Framework.

To facilitate studies in Deep Reinforcement Learning (DRL) and autonomous vehicles, we present the CarAware framework for detailed multi-agent vehicle simulations, which works together with the open-source traffic simulator CARLA. This framework aims to fill the gap identified in currently available CARLA DRL frameworks, often focused on the perception and control of a single vehicle. The new framework provides baselines for training DRL agents in scenarios with multiple connected autonomous vehicles (CAVs), focusing on their sensors' data fusion for objects' localization and identification. These features and tools allow studying many different DRL strategies and algorithms, applied for multi-vehicle sensors' data fusion and interpretation.

A paper describing this framework was submitted to MT-ITS 2023 - 8th International Conference on Models and Technologies for Intelligent Transportation Systems.
After publication, the link to the paper will be available HERE.

# Main Necessary Softwares and Python Libraries
The following software packages/python libraries need to be installed for this framework to work properly:

- Python 3.7
- CARLA Simulator
- Unreal Engine 4
- TensorFlow 1.15
- TensorBoard 1.15
- Numpy
- Gym
- OpenCV
- Open3D

# User Guide
Open "main.py", and edit the SIM_PARAMS, SENS_PARAMS and HYPER_PARAMS according to your simulation and training preferences.
To select the working mode, please change the variable TRAIN_MODE, selecting "Train" to train a new or continue the training of a model, selecting "Play" to run a session with the model indicated in the variable TRAIN_MODEL, or selecting "Simulation" to only run a simulation session without the reinforcement learning implementations.
Once the variables setup is complete, run the "main.py" code to execute the selected scenario. To see the training metrics, execute tensorboard, pointing it to the logs folder of the desired training session, inside the models folder (created upon the first execution of the framework).

# Current known bugs
Some known bugs are present in the initial version of the framework:

- In Town10, if there is no episode reset, the car agent becomes "confused" conducting the vehicle when an episode ends, and tend to crash it.
- Target-Std logic was built to end the training if a certain standard deviation is reached during training, but somehow this training ending is currently not working.

# Future Improvements
Other implementations could also be performed in the CarAware framework, to improve its simulation capabilities and deliver better training strategies:

•	Implement CARLA's synchronous mode support in the framework, useful in the high-processing demanding simulation scenarios. This could also be used to speed-up the simulation, with environments that could run faster than real-time;
•	Implement asynchronous training capabilities in the framework, enabling the simulation of multiple environments, with the possibility of distributed learning over the network for DRL algorithms that are compatible with this feature, like PPO;
•	Implement configurable scenario setup via the top-view window, selecting with mouse clicks the spawn points of each object and their target waypoints, which could enable a better exploration of all maps' regions;
•	Implement a flexible curriculum configuration tool, to create more dynamic curricula, to increase the simulations' randomness in a single episode (which prevents biasing), like dynamic number of vehicles and paths taken, random sensors' blackout, variable sensors' noise, to name a few. Automatic curricula algorithms could also be applied;
•	Implement the structure for other DRL algorithms, like DQN, A2C, A3C, DDPG, and TRPO;
•	Implement other hybrid learning algorithms, that mix DRL with other techniques like imitation learning or genetic algorithms, which present improved results compared to pure DRL algorithms.
•	Implement the interface for other Deep Learning algorithms, focused on sensor fusion studies;
•	Implement a "hero" vehicle option, a vehicle that is completely controlled by the user's script based on its own sensors.
