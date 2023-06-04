import glob
import os
import sys
import time
import carla
#import logging
#import random
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
#import pygame
import math
#import re
#import open3d as o3d
#import top_view
import simulation
#from skimage import transform  # Help us to preprocess the frames
#from collections import deque  # Ordered collection with ends
import train as train_RL
import play as play_RL
from threading import Thread
#from threading import Timer
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU

#from agents.navigation.behavior_agent import BehaviorAgent
#from agents.navigation.basic_agent import BasicAgent

# Desabilita warnings do tensorflow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# CARREGA ARQUIVO .EGG COM MÓDULO PYTHON API DO CARLA
try:
    sys.path.append(glob.glob('C:\carla\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ========== VARIÁVEIS GLOBAIS ==========
SIM_PARAMS = {}

# ========== CONFIG DOS EPISÓDIOS DE SIMULAÇÃO ===========
SIM_PARAMS["EPISODE_RESET"] = True  # Se True, faz o respawn aleatório a cada novo episódio
SIM_PARAMS["SENSORS_BLACKOUT"] = False  # Se True, falha os sensores a cada X segundos, por Y segundos.
SIM_PARAMS["MAP"] = "Town02"  # Mapa que será carregado na simulação. Ex.: Town01,Town02,Town10HD_Opt (só com ep. reset), Random,Gradual_Random
SIM_PARAMS["RANDOM_MAPS"] = ["Town02", "Town10HD_Opt"]  # Mapas que serão selecionados randomicamente se MAP = "Random" ou "Gradual_Random"
SIM_PARAMS["GRADUAL_RANDOM_INIT_EP_CHANGE"] = 100  # Número de episódios que irá rodar no início, antes de trocar o mapa
SIM_PARAMS["GRADUAL_RANDOM_RATE"] = 5  # Tamanho do passo de redução do número de episódios que irá rodar antes de trocar o mapa
SIM_PARAMS["NUM_EPISODES"] = int(0)  # total de episódios que serão rodados (0 or less trains forever)
SIM_PARAMS["EGO_VEHICLE_NUM"] = 1 # Número de Ego vehicles gerados na simulação
SIM_PARAMS["NPC_VEHICLE_NUM"] = 0  # Número de NPC vehicles gerados na simulação
SIM_PARAMS["STATIC_PROPS_NUM"] = 0  # Número de objetos estáticos que serão inseridos no meio da rua
SIM_PARAMS["PEDESTRIAN_NUM"] = 0  # Número de pedestres na simulação
SIM_PARAMS["PERCENTAGE_PEDESTRIANS_RUNNING"] = 0.0  # how many pedestrians will run
SIM_PARAMS["PERCENTAGE_PEDESTRIANS_CROSSING"] = 0.0  # how many pedestrians will walk through the road
# Define o comportamento da direção automática dos carros
SIM_PARAMS["VEHICLE_AGENT"] = "BEHAVIOR"  # Tipo de agente usado no controle dos veículos simulados - Opções: "BEHAVIOR", "BASIC", "SERVER", "STOP"
SIM_PARAMS["VEHICLE_BEHAVIOR"] = "randomized"  # Opções do modo BEHAVIOR: "cautious", "normal", "aggressive", "randomized"
SIM_PARAMS["VEHICLE_DISTANCE"] = 3.0  # Distância de segurança entre veículos, para não baterem
SIM_PARAMS["VEHICLE_SPEED"] = 30  # Define velocidade fixa dos veículos (numeral 0-100) ou se seguem o limite (string "Limit")
SIM_PARAMS["NUM_MIN_WAYPOINTS"] = 20  # Número mínimo de waypoints de destino, caso sejam utilizados (modo behavior)
#OBS: Modo SERVER pesa no servidor, modo BASIC pesa no cliente

# ============= CONFIGURAÇÃO DO CLIMA ===========
SIM_PARAMS["CUSTOM_WEATHER"] = False
# PARÂMETROS CUSTOM
SIM_PARAMS["SUN_ALTITUDE"] = 30
SIM_PARAMS["FOG_DENSITY"] = 0
SIM_PARAMS["FOG_DISTANCE"] = 0
SIM_PARAMS["PRECIPITATION_VALUE"] = 0
SIM_PARAMS["PRECIPITATION_DEPOSITS"] = 0
SIM_PARAMS["CLOUDINESS"] = 0
# PRESETS
SIM_PARAMS["WEATHER_PRESET"] = 2  # 2-Default
# 0-Clear Noon / 1-Clear Sunset / 2-Cloudy Noon / 3-Cloudy Sunset / 4-Default / 5-Hard Rain Noon / 6-Hard Rain Sunset
# 7-Mid Rainy Sunset / 8-Mid Rainy Noon / 9-Soft Rain Noon / 10-Soft Rain Sunset / 11-Wet Cloudy Noon
# 12-Wet Cloudy Sunset / 13-Wet Noon / 14-Wet Sunset

# ============================== CONFIG DO TOP-VIEW ===================================
SIM_PARAMS["TOP_VIEW_SHOW_HUD"] = True  # Habilita exibição do HUD
SIM_PARAMS["TOP_VIEW_SHOW_ID"] = True  # Habilita exibição do ID dos objetos no mapa
SIM_PARAMS["DEBUG"] = True  # Habilita exibição de informações de sensores no HUD (reduz FPS)
SIM_PARAMS["SCREEN_WIDTH"] = 1920  # 1920
SIM_PARAMS["SCREEN_HEIGHT"] = 1020  # 1080
SIM_PARAMS["CONFIG_FPS"] = 30  # Set this to the FPS of the environment
SIM_PARAMS["KALMAN_FILTER"] = False  # Generates kalman filter outputs to compare with the prediction

# ======================== CONFIG DO REINFORCEMENT LEARNING ===========================
SIM_PARAMS["TRAIN_MODE"] = "Play"  # Define o modo de execução do RL: "Train", "Play" ou "Simulation"
SIM_PARAMS["TRAIN_MODEL"] = "Latest"  # "Latest" ou "Nome do modelo" a ser utilizado.
SIM_PARAMS["TRAIN_RESTART"] = False  # Se True, sobrescreve o modelo criado previamente, em False, continua treinamento
SIM_PARAMS["PREDICTION_PREVIEW"] = True  # Se True, desenha a previsão na visão Top-view
SIM_PARAMS["PREDICTION_HUD"] = True  # Se True, insere informações de prediction no HUD
SIM_PARAMS["LAST_POSITIONS_TRAINING"] = False  # Se True, passa as últimas 4 posições para a rede no treinamento

#  Melhores modelos:
#  PPO_MODEL_moving_restart_multi_agent_rw_distance_normalized_step3_v1:
#  "model.ckpt_Interval_14_23_05_eps_-695" / "model.ckpt_Interval_14_42_57_eps_-700"

# ============================= HYPER PARAMETERS ========================================
HYPER_PARAMS = {}
HYPER_PARAMS["learning_rate"] = float(1e-4)  # Initial learning rate - Default: 1e-4 (funcionou) / 5e-4 (ruim) / 8e-5 (devagar) - Erros: 1e-3 gera NaN de output, pesos da NN tendem a infinito
HYPER_PARAMS["lr_decay"] = float(1.0)  # Per-episode exponential learning rate decay - Default: 1.0 (mantêm constante)
HYPER_PARAMS["discount_factor"] = float(0.99)  # GAE discount factor
HYPER_PARAMS["gae_lambda"] = float(0.95)  # GAE lambda
HYPER_PARAMS["ppo_epsilon"] = float(0.2)  # PPO Epsilon - Default: 0.2
HYPER_PARAMS["initial_std"] = float(1.0)  # Initial value of the std used in the gaussian policy - Default: 1.0 (funcionou)
HYPER_PARAMS["target_std"] = int(0.4)  # Target de desvio padrão, utilizado para finalizar treinamento quando atingido - NÃO ESTÁ FUNCIONANDO
HYPER_PARAMS["value_scale"] = float(1.0)  # Value loss scale factor
HYPER_PARAMS["entropy_scale"] = float(0.01)  # Entropy loss scale factor - Default: 0.01
HYPER_PARAMS["horizon"] = int(32768)  # Number of steps to simulate per training step - Default: 128 / 32768 (funcionou)
HYPER_PARAMS["num_training"] = int(1)  # Number of times the model will be trained per episode
HYPER_PARAMS["num_epochs"] = int(4)  # Number of PPO training epochs per traning step - Default: 3 (funcionou) / 4
HYPER_PARAMS["batch_size"] = int(2048)  # Epoch batch size - Default: 32 / 2048 (funcionou) / 8192
HYPER_PARAMS["synchronous"] = False  # Set this to True when running in a synchronous environment
HYPER_PARAMS["action_smoothing"] = float(0.0)  #Action smoothing factor
HYPER_PARAMS["model_name"] = "Scenario1play"  # Name of the model to train. Output written to models/model_name
HYPER_PARAMS["reward_fn"] = "rw_distance_normalized"  # Reward Function to use. See reward_functions.py for more info.
HYPER_PARAMS["seed"] = 0  # Seed to use. (Note that determinism unfortunately appears to not be guaranteed
                        # with this option in our experience)
HYPER_PARAMS["eval_interval"] = int(10)  # Number of episodes interval between evaluations - Default: 5
#HYPER_PARAMS["save_eval_interval"] = int(10)  # Number of evaluations interval for saving (in addition to best rw)
HYPER_PARAMS["eval_time"] = int(30)  # Tempo que a simulação irá rodar para avaliação - Default: 60
HYPER_PARAMS["record_eval"] = True  # If True, save' videos of evaluation episodes to models/model_name/videos/
# HYPER_PARAMS["reset_mode"] = RESET_MODE  # Usado em conjunto com restart, define se reinicia sempre ou só target

# =========== CONFIGURAÇÃO DOS SENSORES ( HABILITAÇÃO É True ou False) ============================
SENS_PARAMS = {}
# SPEED AND STEERING ANGLE SENSOR (SPD_SAS) - Funciona apenas com carro em movimento
SENS_PARAMS["SENS_SPD_SAS"] = True
SENS_PARAMS["SENS_SPD_SAS_SAMPLING"] = 0.1  # tempo em segundos entre cada aquisição
SENS_PARAMS["SENS_SPD_SAS_ERROR"] = 0.01  # Default: 0.001
SENS_PARAMS["SENS_SPD_SAS_BLACKOUT_ON"] = False  # Habilita/desabilita blackout desse sensor
SENS_PARAMS["SENS_SPD_SAS_BLACKOUT_MIN"] = 5  # Tempo em segundos que o sensor ficará desabilitado a cada X períodos.
SENS_PARAMS["SENS_SPD_SAS_BLACKOUT_MAX"] = 10
SENS_PARAMS["SENS_SPD_SAS_BLACKOUT_INTERVAL_MIN"] = 5  # Tempo em segundos do intervalo de blackout
SENS_PARAMS["SENS_SPD_SAS_BLACKOUT_INTERVAL_MAX"] = 10

# GLOBAL NAVIGATION SATELLITE SYSTEM (GNSS)
SENS_PARAMS["SENS_GNSS"] = True
SENS_PARAMS["SENS_GNSS_PREVIEW"] = True  # Define se os pontos detectados serão desenhados na tela
SENS_PARAMS["SENS_GNSS_SAMPLING"] = 0.1  # tempo em segundos entre cada aquisição
SENS_PARAMS["SENS_GNSS_ERROR"] = 0.00005  # Default: Low = 0.00001 / High = 0.0001
SENS_PARAMS["SENS_GNSS_BIAS"] = 0.0
SENS_PARAMS["SENS_GNSS_BLACKOUT_ON"] = True  # Habilita/desabilita blackout desse sensor
SENS_PARAMS["SENS_GNSS_BLACKOUT_MIN"] = 5  # Tempo em segundos que o sensor ficará desabilitado a cada X períodos.
SENS_PARAMS["SENS_GNSS_BLACKOUT_MAX"] = 10
SENS_PARAMS["SENS_GNSS_BLACKOUT_INTERVAL_MIN"] = 5  # Tempo em segundos do intervalo de blackout
SENS_PARAMS["SENS_GNSS_BLACKOUT_INTERVAL_MAX"] = 10

# INERTIAL MEASUREMENT UNIT (IMU)
SENS_PARAMS["SENS_IMU"] = True
SENS_PARAMS["SENS_IMU_SAMPLING"] = 0.1  # tempo em segundos entre cada aquisição
SENS_PARAMS["SENS_IMU_ACCEL_ERROR"] = 0.001  # Default: 0.00001
SENS_PARAMS["SENS_IMU_GYRO_ERROR"] = 0.001  # Default: 0.00001
SENS_PARAMS["SENS_IMU_GYRO_BIAS"] = 0.0
SENS_PARAMS["SENS_IMU_BLACKOUT_ON"] = False  # Habilita/desabilita blackout desse sensor
SENS_PARAMS["SENS_IMU_BLACKOUT_MIN"] = 5  # Tempo em segundos que o sensor ficará desabilitado a cada X períodos.
SENS_PARAMS["SENS_IMU_BLACKOUT_MAX"] = 10
SENS_PARAMS["SENS_IMU_BLACKOUT_INTERVAL_MIN"] = 5  # Tempo em segundos do intervalo de blackout
SENS_PARAMS["SENS_IMU_BLACKOUT_INTERVAL_MAX"] = 10

# COLLISION DETECTION (COL)
SENS_PARAMS["SENS_COL"] = False

# OBSTACLE DETECTION (OBS)
SENS_PARAMS["SENS_OBS"] = False

# CAMERA DE VÍDEO A CORES (RGB)
SENS_PARAMS["SENS_RGB"] = False
SENS_PARAMS["SENS_RGB_PREVIEW"] = False  # Define se as imagens captadas serão desenhadas na tela
SENS_PARAMS["SENS_RGB_SAMPLING"] = 3  # tempo em segundos entre cada aquisição
SENS_PARAMS["SENS_RGB_STACK_SIZE"] = 4  # define o tamanho do buffer com X imagens para alimentar a RN
SENS_PARAMS["RGB_MODE"] = "SEMANTIC"  # Valores possíveis: YOLO, BINARY, SEMANTIC
SENS_PARAMS["IM_WIDTH"] = 320  # 640   160
SENS_PARAMS["IM_HEIGHT"] = 160  # 480   80
SENS_PARAMS["SENS_RGB_BLACKOUT"] = 0  # Tempo em segundos que o sensor ficará desabilitado a cada X períodos. 0 = blackout desativado
SENS_PARAMS["SENS_RGB_BLACKOUT_INTERVAL"] = 10  # Tempo em segundos do intervalo de blackout

# LIGHT DETECTION AND RANGING (LIDAR)
SENS_PARAMS["SENS_LIDAR"] = False
SENS_PARAMS["SENS_LIDAR_PREVIEW"] = False  # Define se os pontos detectados serão desenhados na tela
SENS_PARAMS["SENS_LIDAR_SAMPLING"] = 0.3  # Default 0
SENS_PARAMS["SENS_LIDAR_RANGE"] = 20  # 20
SENS_PARAMS["SENS_LIDAR_NUM_POINTS"] = 90000  # 90000
SENS_PARAMS["SENS_LIDAR_FREQUENCY"] = 20  # 40
SENS_PARAMS["SENS_LIDAR_CHANNELS"] = 32  # 32
SENS_PARAMS["SENS_LIDAR_SHOW_FACTOR"] = 10  # 10
SENS_PARAMS["SENS_LIDAR_TOP_VIEW"] = "INTEREST"  # valores possíveis: ALL, INTEREST
SENS_PARAMS["SENS_LIDAR_BLACKOUT"] = 0  # Tempo em segundos que o sensor ficará desabilitado a cada X períodos. 0 = blackout desativado
SENS_PARAMS["SENS_LIDAR_BLACKOUT_INTERVAL"] = 10  # Tempo em segundos do intervalo de blackout

# ======== CORES P/ POINT CLOUD SEMÂNTICO ========
SENS_PARAMS["LABEL_COLORS"] = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
])  # / 255.0 # normalize each channel [0-1] since is what Open3D uses

# CORES PARA BOUNDING BOXES GERADAS PELO YOLO (CÂMERA RGB)
SENS_PARAMS["YOLO_COLORS"] = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


# ===== PROGRAMA PRINCIPAL =====
def main():

    # UNPACK DAS VIARIÁVEIS UTILIZADAS NESSE PROGRAMA
    MAP = SIM_PARAMS["MAP"]
    TRAIN_MODE = SIM_PARAMS["TRAIN_MODE"]
    EPISODE_RESET = SIM_PARAMS["EPISODE_RESET"]
    EGO_VEHICLE_NUM = SIM_PARAMS["EGO_VEHICLE_NUM"]
    NPC_VEHICLE_NUM = SIM_PARAMS["NPC_VEHICLE_NUM"]
    STATIC_PROPS_NUM = SIM_PARAMS["STATIC_PROPS_NUM"]
    PEDESTRIAN_NUM = SIM_PARAMS["PEDESTRIAN_NUM"]
    PREDICTION_HUD = SIM_PARAMS["PREDICTION_HUD"]
    DEBUG = SIM_PARAMS["DEBUG"]
    VEHICLE_AGENT = SIM_PARAMS["VEHICLE_AGENT"]
    TOP_VIEW_SHOW_HUD = SIM_PARAMS["TOP_VIEW_SHOW_HUD"]
    NUM_MIN_WAYPOINTS = SIM_PARAMS["NUM_MIN_WAYPOINTS"]

    SENS_GNSS = SENS_PARAMS["SENS_GNSS"]
    SENS_IMU = SENS_PARAMS["SENS_IMU"]
    SENS_SPD_SAS = SENS_PARAMS["SENS_SPD_SAS"]
    SENS_OBS = SENS_PARAMS["SENS_OBS"]
    SENS_COL = SENS_PARAMS["SENS_COL"]
    SENS_RGB = SENS_PARAMS["SENS_RGB"]
    SENS_LIDAR = SENS_PARAMS["SENS_LIDAR"]
    SENS_RGB_PREVIEW = SENS_PARAMS["SENS_RGB_PREVIEW"]

    # INICIALIZA AS CLASSES DA SIMULAÇÃO
    sim = simulation.SimulationSetup(SIM_PARAMS, SENS_PARAMS)  # Classe com setup da simulação
    sim.simulation_status = "Loading"  # Informa que a simulação está sendo carregada

    sim_pause = simulation.SimPause()  # Classe que pausa/resume a simulação
    sim_pause.start(sim)
    sim_pause.pause()  # pausa a simulação para configuração do primeiro episódio

    top_view = simulation.TopView(SIM_PARAMS)  # Classe que abre a janela de Top View
    if MAP != ("Random" or "Gradual_Random"):
        top_view.start(MAP)
    #top_view.start()

    if TRAIN_MODE == "Train":  # inicializa thread p/ treinamento RL
        trainer_thread = Thread(target=train_RL.train, args=(HYPER_PARAMS, SIM_PARAMS, sim, top_view), daemon=True)
        trainer_thread.start()
    if TRAIN_MODE == "Play":  # inicializa thread p/ preview de modelo treinado RL
        trainer_thread = Thread(target=play_RL.play, args=(HYPER_PARAMS, SIM_PARAMS, sim, top_view), daemon=True)
        trainer_thread.start()
        sim.simulation_status = "Play_Loading"
    if TRAIN_MODE == "Simulation":
        sim.simulation_status = "Simulation"

    # PREENCHE STRING SENSORES COM OS QUE ESTÃO HABILITADOS P/ MOSTRAR NO HUD
    sensores = ""
    sensores += "GNSS " if SENS_GNSS else ""
    sensores += "IMU " if SENS_IMU else ""
    sensores += "SPD/SAS" if SENS_SPD_SAS else ""
    sensores += "OBS " if SENS_OBS else ""
    sensores += "COL " if SENS_COL else ""
    sensores += "RGB " if SENS_RGB else ""
    sensores += "LIDAR " if SENS_LIDAR else ""

    num_restarts = 0  # contabiliza número de vezes que reiniciou
    sim_start_time = time.time()

    First_episode = True  # Faz o spawn no primeiro episódio simulado
    while not sim.simulation_status == "Complete":
    #for episode_num in range(EPISODE_TOTAL):

        sim.new_episode = True
        #simulation.episodio_atual +=1

        # Inicia exibição de dados do mapa selecionado na tela de top-view
        if MAP == "Random" or MAP == "Gradual_Random":
            top_view.start(sim.chosen_random_map)
        #else:
        #    top_view.start(MAP)

        # Determina se o número de episódios é infinito ou não
        if SIM_PARAMS["NUM_EPISODES"] == 0:
            num_episodes = math.inf
        else:
            num_episodes = str(SIM_PARAMS["NUM_EPISODES"])

        # INICIALIZA O EPISÓDIO "EPISODE_NUM"
        # registra os eventos em formato de log
        print("\n======= Iniciando episódio", sim.episodio_atual, "DE", num_episodes, "=======\n")
        if MAP == "Random":
            print("Mapa selecionado: ", sim.chosen_random_map)

        # CARREGA PEDESTRES, VEÍCULOS E OBJETOS
        if EPISODE_RESET or First_episode == True or sim.simulation_reset == True:
            sim.spawn_all()
            First_episode = False

        # CONFIGURA O EXPECTADOR PARA VISÃO DE CIMA NO SERVIDOR
        spectator = sim.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(105.462921, 96.121056, 129.900925),
                                                carla.Rotation(-59.038227, 90.226158, 0.001122)))

        #top_view.world.ground_truth()  # gera os dados de GT para o RL

        # SINALIZA SIMULAÇÃO OK P/ TREINAMENTO COMEÇAR
        if sim.simulation_status != "Play_Loading" and sim.simulation_status != "Simulation":
            sim.simulation_status = "Ready"
            # registra os eventos em formato de log
            print("\nEpisódio Iniciado - Rodando por", str(HYPER_PARAMS["horizon"]), "horizontes")

        sim_pause.resume()  # RESUME A SIMULAÇÃO APÓS A CONFIGURAÇÃO

        # Game loop
        ep_start_time = time.time()

        # Espera treinamento começar
        while not (sim.simulation_status == "Training" or sim.simulation_status == "Play" or \
                sim.simulation_status == "Simulation"):
            time.sleep(0.01)

        # while time.time() <= ep_start_time + EPISODE_TIME:
        while sim.simulation_status == "Training" or sim.simulation_status == "Play" or \
                sim.simulation_status == "Simulation":

            # Lógica para finalizar com ESC no modo simulação
            if sim.simulation_status == "Simulation" and top_view.input_control.quit :
                sim.simulation_status = "Complete"
                break

            #top_view.world.ground_truth()  # gera os dados de GT para o RL
            hud_txt = []

            if TOP_VIEW_SHOW_HUD:  # mostra dados no HUD do modo Top-view
                sim.sim_total_time = round((time.time() - sim_start_time)+sim.sim_last_total_time)
                hud_txt.append("SIMULAÇÃO_Modo: %s;Episódio: %s / %s;Restarts: %s;Treinamento: %s / %s;Horizonte: %s / %s ;Tempo de simulação: %s;Tempo do episódio: %s s;"
                           "Número de carros EGO: %s;Número de carros NPC: %s;Número de obstáculos: %s;Número de pedestres %s;Sensores: %s"
                           % (
                               str(sim.simulation_status),
                               str(sim.episodio_atual), num_episodes,
                               num_restarts,
                               str(sim.training_atual+1), str(HYPER_PARAMS["num_training"]),
                               sim.horizonte_atual + 1, HYPER_PARAMS["horizon"],
                               '{:02}:{:02}:{:02}'.format(sim.sim_total_time // 3600, sim.sim_total_time % 3600 // 60, sim.sim_total_time % 60),
                               #str(format(time.time() - sim_start_time, ".2f")),
                               str(round(time.time() - ep_start_time)),
                               str(EGO_VEHICLE_NUM), str(NPC_VEHICLE_NUM), str(STATIC_PROPS_NUM),
                               str(PEDESTRIAN_NUM), sensores))

            if PREDICTION_HUD:  # exibe informações de prediction/reward no HUD

                # Informações de rewards
                predictions = "PREDICTIONS_Reward Eval. Max.: %s;Reward Eval. Atual: %s;Reward Instantâneo: %s;" % (str('{:.2f}'.format(sim.best_reward)),
                                                                                   str('{:.2f}'.format(sim.reward_atual)), str('{:.2f}'.format(sim.reward_inst)))

                # Informações de predictions
                for idx, veh in zip(range(len(sim.ego_vehicle)), sim.ego_vehicle):
                    if veh.pred_distance is not None:
                        predictions += "Carro " + str(idx+1) + ": " + str('{:.2f}'.format(veh.pred_distance)) + ";"

                        #predictions += "CARRO " + str(idx) + ": [" + str(int(veh.prediction[0])) + ", " + \
                        #           str(int(veh.prediction[1])) + ", " + str(int(veh.prediction[2])) + "];"
                hud_txt.append(predictions)

            # ============ CONTROLE DA SIMULAÇÃO DOS EGO VEHICLES ============
            if DEBUG or sim.simulation_status == "Play":  # Exibe apenas se modo DEBUG estiver habilitado
                car_info = "SENSORES_"
            for veh in sim.ego_vehicle:
                # CONTROLE DE COMPORTAMENTO DA DIREÇÃO AUTOMÁTICA
                if VEHICLE_AGENT == "BASIC":
                    control = veh.agent.run_step()
                    control.manual_gear_shift = False
                    veh.apply_control(control)
                elif VEHICLE_AGENT == "BEHAVIOR":
                    #veh.agent.update_information()
                    #if len(veh.agent.get_local_planner().waypoints_queue) < NUM_MIN_WAYPOINTS:
                        #veh.agent.reroute(simulation.veh_spawn_points)
                    speed_limit = veh.get_speed_limit()
                    veh.agent.get_local_planner().set_speed(speed_limit)
                    control = veh.agent.run_step()
                    veh.apply_control(control)

                # MOSTRA DADOS DOS VEÍCULOS NO HUD, CASO TOP-VIEW ESTEJA HABILITADO
                if TOP_VIEW_SHOW_HUD:  # mostra dados no HUD do modo Top-view
                    if DEBUG or sim.simulation_status == "Play":  # habilita exibição das leituras dos sensores no hud (reduz FPS)
                        try:
                            car_info += 'CARRO "%s";' % (veh.attributes["role_name"])
                            car_info += "Velocidade: %13.0f km/h;" % (veh.sens_spd_sas_speed)
                            car_info += "Âng. Volante: %17.2fº;" % (veh.sens_spd_sas_angle)
                            if SENS_IMU and veh.sens_imu is not None:
                                car_info += "Direção: %17.2fº %2s;" % (
                                    veh.sens_imu.ue_compass_degrees, veh.sens_imu.ue_compass_heading)
                                car_info += "Acelerômetro:  (%4.1f,%4.1f,%4.1f);" % (veh.sens_imu.ue_accelerometer[0],
                                                                             veh.sens_imu.ue_accelerometer[1],
                                                                             veh.sens_imu.ue_accelerometer[2])
                                car_info += "Giroscópio:    (%4.1f,%4.1f,%4.1f);" % (veh.sens_imu.ue_gyroscope[0],
                                                                             veh.sens_imu.ue_gyroscope[1],
                                                                             veh.sens_imu.ue_gyroscope[2])
                        except:
                            pass

                    if SENS_RGB and veh.sens_rgb is not None:
                        try:
                            #car_info += "Detecção da câmera: %s;" % veh.sens_rgb_objid
                            if SENS_RGB_PREVIEW:
                                cv2.imshow("CARRO " + str(veh.attributes["role_name"]), veh.sens_rgb_data)
                                cv2.waitKey(1)
                        except:
                            pass

            # ============ CONTROLE DA SIMULAÇÃO DOS NPC VEHICLES ============
            for veh in sim.npc_vehicle:
                if VEHICLE_AGENT == "BASIC":
                    control = veh.agent.run_step()
                    control.manual_gear_shift = False
                    veh.apply_control(control)
                elif VEHICLE_AGENT == "BEHAVIOR":
                    veh.agent.update_information()
                    if len(veh.agent.get_local_planner().waypoints_queue) < NUM_MIN_WAYPOINTS:
                        veh.agent.reroute(sim.veh_spawn_points)
                    speed_limit = veh.get_speed_limit()
                    veh.agent.get_local_planner().set_speed(speed_limit)
                    control = veh.agent.run_step()
                    veh.apply_control(control)
            if TOP_VIEW_SHOW_HUD:  # mostra dados no HUD do modo Top-view
                if DEBUG or sim.simulation_status == "Play":
                    hud_txt.append(car_info)
            top_view.tick(hud_txt, sim.ego_vehicle)  # atualiza a exibição do top-view

        sim_pause.pause()  # PAUSA A SIMULAÇÃO PARA CONFIGURAR O EPISÓDIO

        if EPISODE_RESET or sim.simulation_reset == True:
            sim.reset()
            sim.simulation_reset = False
            num_restarts += 1

        if sim.simulation_status != "Complete":
            sim.simulation_status = "Loading"  # Segura o treinamento enquanto a simulação reinicia



        # registra os eventos em formato de log
        print("Episódio Finalizado")

    # APAGA OS OBJETOS DE SIMULAÇÃO CRIADOS
    del sim
    # registra os eventos em formato de log
    print("\nSimulação finalizada!")
    return top_view.input_control.quit_reason

if __name__ == '__main__':

    while True:

        try:
            subprocess.call('taskkill /f /fi "IMAGENAME eq CarlaUE4*"', shell=True)
            pass
        except:
            pass

        os.startfile("C:\carla\CarlaUE4.exe") # ABRE O CARLA
        time.sleep(5)
        #subprocess.call(["C:\carla13\CarlaUE4.exe","-fps=5"])
        result = main()

        if result == "Crash":
            print("Crash detectado no CarlaUE4, reiniciando simulação para continuar treinamento")
        else:  # Crash
            print("Simulação encerrada através da tecla ESC")
            break

    subprocess.call('taskkill /f /fi "IMAGENAME eq CarlaUE4*"', shell=True)

    sys.exit()