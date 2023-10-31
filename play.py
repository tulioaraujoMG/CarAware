import os
import random

import numpy as np
import tensorflow as tf

from ppo import PPO
#from reward_functions import reward_functions
from run_eval import run_eval


from CarlaEnv.carla_env import CarlaEnv as CarlaEnv


def play(hyper_params, sim_params, simulation, top_view):  # start_carla=True
    # Read parameters
    learning_rate = hyper_params["learning_rate"]
    lr_decay = hyper_params["lr_decay"]
    discount_factor = hyper_params["discount_factor"]
    gae_lambda = hyper_params["gae_lambda"]
    ppo_epsilon = hyper_params["ppo_epsilon"]
    initial_std = hyper_params["initial_std"]
    value_scale = hyper_params["value_scale"]
    entropy_scale = hyper_params["entropy_scale"]
    horizon = hyper_params["horizon"]
    num_training = hyper_params["num_training"]
    num_epochs = hyper_params["num_epochs"]
    num_episodes = sim_params["NUM_EPISODES"]
    batch_size = hyper_params["batch_size"]
    # vae_model        = params["vae_model"]
    # vae_model_type   = params["vae_model_type"]
    # vae_z_dim        = params["vae_z_dim"]
    synchronous = hyper_params["synchronous"]
    fps = sim_params["CONFIG_FPS"]
    action_smoothing = hyper_params["action_smoothing"]
    model_name = hyper_params["model_name"]
    reward_fn = hyper_params["reward_fn"]
    seed = hyper_params["seed"]
    eval_interval = hyper_params["eval_interval"]
    # save_eval_interval=params["save_eval_interval"]
    record_eval = hyper_params["record_eval"]
    ego_num = sim_params["EGO_VEHICLE_NUM"]
    # reset = sim_params["EPISODE_RESET"]
    # vehicle_agent = sim_params["VEHICLE_AGENT"]
    # restart = sim_params["TRAIN_RESTART"]
    # reset_mode       = params["reset_mode"]
    train_model = sim_params["TRAIN_MODEL"]
    # target_std = hyper_params["target_std"]
    map = sim_params["MAP"]
    record_play_stats = sim_params["RECORD_PLAY_STATS"]
    # last_positions_training = sim_params["LAST_POSITIONS_TRAINING"]
    eval_time        = 999999999  # roda por tempo indefinido até ESC ser pressionado
    simulation.eval = True

    # Set seeds
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(0)

    # Create env
    print("Creating environment")
    env = CarlaEnv(#obs_res=(160, 80),
                   action_smoothing=action_smoothing,
                   #encode_state_fn=encode_state_fn,
                   reward_fn=reward_fn,
                   synchronous=synchronous,
                   fps=fps,
                   #start_carla=start_carla
                   simulation=simulation, top_view=top_view,
                   ego_num=ego_num,
                   map=map)

    if isinstance(seed, int):
        env.seed(seed)
    best_eval_reward = -float("inf")

    # Environment constants
    input_shape = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    #input_shape = env.observation_space["GNSS"].shape[0] + 1  # input_shape = np.array([vae.z_dim + len(measurements_to_include)])
    #num_actions = env.action_space["Obj_Coord"].shape[0] + 1 # antes era +1

    # Create model
    print("Creating model")
    model = PPO(input_shape, env.action_space,
                learning_rate=learning_rate, lr_decay=lr_decay,
                epsilon=ppo_epsilon, initial_std=initial_std,
                value_scale=value_scale, entropy_scale=entropy_scale,
                model_dir=os.path.join("models", model_name))

    model.init_session()

    if train_model == "Latest":
        model.load_latest_checkpoint()
    else:  # Custom model
        model.load_custom_checkpoint(train_model)

    print("Rodando em modo PREVIEW com modelo: \"{}\".".format(model_name))

    simulation.simulation_status = "Play"
    play = True
    run_eval(env, model, None, eval_time, simulation, ego_num, play, record_play_stats)

    #eval_reward = run_eval(env, model, eval_time=eval_time)
    #print("Reward final: ", eval_reward)

    # Reprodução finalizada
    simulation.simulation_status = "Complete"