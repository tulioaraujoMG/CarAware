#import os
#import random
#import re
#import shutil
import time
#from collections import deque

import cv2
#import gym
#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
#from skimage import transform

#from PIL import Image
#from ppo import PPO
#from vae.models import ConvVAE, MlpVAE
#from CarlaEnv.wrappers import angle_diff, vector
from utils import VideoRecorder, ExtendedKalmanFilter
#from vae_common import create_encode_state_fn, load_vae
#from reward_functions import reward_functions
#import pygame
import pyautogui
import pygetwindow as gw

#from CarlaEnv.carla_env import CarlaEnv as CarlaEnv


def run_eval(env, model, video_filename=None, eval_time = 20, simulation = None, ego_num = 0, play = False, record_play_stats = False):

    # Init test env
    state, terminal, total_reward_pred = env.reset(is_training=False) #, False, 0

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
    distance_kf = [0 for i in range(ego_num)]

    # Configuração inicial Extended Kalman Filter
    dt = 0.1
    var_pos = 0.1
    var_acc = 1.0
    ekf = ExtendedKalmanFilter()

    # Define a posição inicial e a aceleração inicial do veículo - Kalman Filter
    #posX = random.uniform(-5, 5)
    #posY = random.uniform(-5, 5)
    #acelX = random.uniform(-2, 2)
    #acelY = random.uniform(-2, 2)

    #Acumula resultados de distãncia para cálculo da distância média prediction RL e KF por veículo
    dist_acum_rl = [[] for i in range(ego_num)]
    dist_acum_kf = [[] for i in range(ego_num)]

    while time.time() < time_now + eval_time:  # X segundos de avaliação

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
        total_reward_pred += reward

        # Lógica para voltar a ciclagem de treinamento para o primeiro veículo

        if current_veh == ego_num - 1:
            current_veh = 0
        else:
            current_veh += 1

        # Registra distância RL
        distance[current_veh] = env.distance
        dist_acum_rl[current_veh].append(distance[current_veh])

        # =============== Lógica Extended Kalman Filter =================

        # Coleta dados GNSS
        posX = simulation.ego_vehicle[current_veh].sens_gnss_input.x
        posY = simulation.ego_vehicle[current_veh].sens_gnss_input.y
        posZ = simulation.ego_vehicle[current_veh].sens_gnss_input.z
        gnss = [posX, posY, posZ]

        # Inicializa EKF com dados GNSS
        ekf.initialize_with_gnss(gnss)

        #z = np.array([[posX], [posY]])
        #acelX = simulation.ego_vehicle[current_veh].sens_imu.ue_accelerometer[0]
        #acelY = simulation.ego_vehicle[current_veh].sens_imu.ue_accelerometer[1]

        # Coleta dados IMU
        imu = simulation.ego_vehicle[current_veh].sens_imu

        # EKF prediction com dados IMU
        ekf.predict_state_with_imu(imu)

        # EKF correction com dados GNSS
        ekf.correct_state_with_gnss(gnss)

        # Get EKF estimated location
        prediction = ekf.get_location()

        # Executa o filtro de Kalman para prever a posição do veículo
        #predX, predY = kf.run(z)

        # Grava valor de predição o filtro de kalman para desenhar pontos na visão top-view
        simulation.ego_vehicle[current_veh].pred_kalman_x = prediction[0]
        simulation.ego_vehicle[current_veh].pred_kalman_y = prediction[1]

        # Calcula distância prediction KF para GT
        veh_gt = env._top_view.world.gt_input_ego

        try:
            distance_kf[current_veh] = np.sqrt((prediction[0] - veh_gt[current_veh].x) ** 2 + (
                    prediction[1] - veh_gt[current_veh].y) ** 2)
            dist_acum_kf[current_veh].append(distance_kf[current_veh])
        except:
            pass


        # Imprime a posição medida, a posição prevista e a aceleração atual
        #print(
        #    f"Medição: ({posX:.2f}, {posY:.2f}) - Predição: ({predX[0]:.2f}, {predY[0]:.2f}) - Aceleração: ({acelX:.2f}, {acelY:.2f})")

        if play and record_play_stats:
            model.write_value_to_summary("play/dist_predictions_veh{}".format(current_veh + 1), distance[current_veh], time.time())
            model.write_value_to_summary("play/dist_kf_veh{}".format(current_veh + 1), distance_kf[current_veh], time.time())


    # Calcula média das distâncias por veículo
    avg_dist_rl_lst = []
    avg_dist_kf_lst = []
    for veh in range(ego_num):
        avg_dist_rl_lst.append(np.mean(dist_acum_rl[veh])) # Reinforcement Learning
        avg_dist_kf_lst.append(np.mean(dist_acum_kf[veh])) # Kalman Filter
        simulation.ego_vehicle[veh].pred_kalman_x = 9999  # some ponto da predição kalman quando não está em eval
        simulation.ego_vehicle[veh].pred_kalman_y = 9999
    avg_dist_rl = np.mean(avg_dist_rl_lst)
    #print(avg_dist_rl)
    avg_dist_kf = np.mean(avg_dist_kf_lst)
    #print(avg_dist_kf)
    #avg_dist = {'avg_dist_rl': avg_dist_rl, 'avg_dist_kf': avg_dist_kf}

    # Escreve resultados no tensorboard
    if not play:
        model.write_value_to_summary("eval/avg_dist_predictions", avg_dist_rl, simulation.episodio_atual)
        model.write_value_to_summary("eval/avg_dist_kf", avg_dist_kf, simulation.episodio_atual)


    #for veh in range(ego_num):
    #    model.write_value_to_summary("eval/distance_veh{}".format(veh + 1),
    #                                     distance[current_veh], simulation.episodio_atual)

    # Release video
    if video_recorder is not None:
        video_recorder.release()

    return total_reward_pred