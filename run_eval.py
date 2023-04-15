import os
import random
import re
import shutil
import time
from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import transform

from PIL import Image
from ppo import PPO
from vae.models import ConvVAE, MlpVAE
from CarlaEnv.wrappers import angle_diff, vector
from utils import VideoRecorder, compute_gae
from vae_common import create_encode_state_fn, load_vae
#from reward_functions import reward_functions
import pygame
import pyautogui
import pygetwindow as gw

#from CarlaEnv.carla_env import CarlaEnv as CarlaEnv


def run_eval(env, model, video_filename=None, eval_time = 20, simulation = None, ego_num = 0):

    # Init test env
    state, terminal, total_reward = env.reset(is_training=False) #, False, 0

    #data = pygame.image.tostring(top_view.display, "RGB", False)
    #pil_img = Image.frombytes("RGB",pygame.display.get_surface().get_size(), data)
    #rendered_frame = np.asarray(pil_img)

    # Init video recording
    if video_filename is not None:
        # get window
        window_name = "Visão Superior"
        w = gw.getWindowsWithTitle(window_name)[0]
        # make a screenshot
        img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        rendered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print("Recording video to {} ({}x{}@{}fps)".format(video_filename, *tuple(w.size), 20))  #int(env.average_fps)
        video_recorder = VideoRecorder(video_filename,
                                       frame_size=tuple(w.size),
                                       fps=20)  #env.average_fps
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    # While non-terminal state
    time_now = time.time()
    #while not terminal:
    current_veh = 0

    distance = [0 for i in range(ego_num)]
    while time.time() < time_now + eval_time:  # 20s de avaliação

        #time.sleep(1/30)
        # Take deterministic actions at test time (std=0)
        action, _ = model.predict(state, greedy=True)

        state, reward, terminal = env.step(action, simulation.ego_vehicle[current_veh], current_veh)

        if video_recorder is not None:
            # make a screenshot
            img = pyautogui.screenshot(region=(w.left, w.top, w.width, w.height))
            # convert these pixels to a proper numpy array to work with OpenCV
            frame = np.array(img)
            # convert colors from BGR to RGB
            rendered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_recorder.add_frame(rendered_frame)
        total_reward += reward

        # Lógica para voltar a ciclagem de treinamento para o primeiro veículo

        if current_veh == ego_num - 1:
            current_veh = 0
        else:
            current_veh += 1

        distance[current_veh] = env.distance

    for veh in range(ego_num):
        model.write_value_to_summary("eval/distance_veh{}".format(veh + 1),
                                         distance[current_veh], simulation.episodio_atual)

    # Release video
    if video_recorder is not None:
        video_recorder.release()

    return total_reward