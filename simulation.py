import carla
import random
import cv2
import numpy as np
import re
from skimage import transform  # Help us to preprocess the frames
from collections import deque  # Ordered collection with ends
import open3d as o3d
import time
import math
from threading import Timer
import logging
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent
import pygame
import top_view

class SimulationSetup:

    # ===== INICIALIZA OS OBJETOS DA SIMULAÇÃO =====
    def __init__(self, sim_params, sens_params):
        # UNPACK DE VARIÁVEIS
        self.ego_vehicle_num = sim_params["EGO_VEHICLE_NUM"]
        self.npc_vehicle_num = sim_params["NPC_VEHICLE_NUM"]
        self.static_props_num = sim_params["STATIC_PROPS_NUM"]
        self.map = sim_params["MAP"]
        self.random_maps = sim_params["RANDOM_MAPS"]
        self.gradual_random_init_ep_change = sim_params["GRADUAL_RANDOM_INIT_EP_CHANGE"]
        self.sensors_blackout = sim_params["SENSORS_BLACKOUT"]
        self.gradual_random_rate = sim_params["GRADUAL_RANDOM_RATE"]
        self.vehicle_distance = sim_params["VEHICLE_DISTANCE"]
        self.custom_weather = sim_params["CUSTOM_WEATHER"]
        self.sun_altitude = sim_params["SUN_ALTITUDE"]
        self.precipitation_value = sim_params["PRECIPITATION_VALUE"]
        self.precipitation_deposits = sim_params["PRECIPITATION_DEPOSITS"]
        self.cloudiness = sim_params["CLOUDINESS"]
        self.fog_density = sim_params["FOG_DENSITY"]
        self.fog_distance = sim_params["FOG_DISTANCE"]
        self.weather_preset = sim_params["WEATHER_PRESET"]
        self.pedestrian_num = sim_params["PEDESTRIAN_NUM"]
        self.percentage_pedestrians_running = sim_params["PERCENTAGE_PEDESTRIANS_RUNNING"]
        self.percentage_pedestrians_crossing = sim_params["PERCENTAGE_PEDESTRIANS_CROSSING"]
        self.vehicle_agent = sim_params["VEHICLE_AGENT"]
        self.vehicle_speed = sim_params["VEHICLE_SPEED"]
        self.vehicle_behavior = sim_params["VEHICLE_BEHAVIOR"]
        self.prediction_preview = sim_params["PREDICTION_PREVIEW"]
        self.centralized_spawn = sim_params["CENTRALIZED_SPAWN"]

        self.rgb_mode = sens_params["RGB_MODE"]
        self.sens_obs = sens_params["SENS_OBS"]
        self.sens_col = sens_params["SENS_COL"]
        self.sens_imu = sens_params["SENS_IMU"]
        self.sens_gnss = sens_params["SENS_GNSS"]
        self.sens_rgb = sens_params["SENS_RGB"]
        self.sens_spd_sas = sens_params["SENS_SPD_SAS"]
        self.sens_lidar = sens_params["SENS_LIDAR"]
        self.im_height = sens_params["IM_HEIGHT"]
        self.im_width = sens_params["IM_WIDTH"]
        self.yolo_colors = sens_params["YOLO_COLORS"]
        self.sens_gnss_sampling = sens_params["SENS_GNSS_SAMPLING"]
        self.sens_gnss_preview = sens_params["SENS_GNSS_PREVIEW"]
        self.sens_gnss_blackout_on = sens_params["SENS_GNSS_BLACKOUT_ON"]
        self.sens_gnss_blackout_min = sens_params["SENS_GNSS_BLACKOUT_MIN"]
        self.sens_gnss_blackout_max = sens_params["SENS_GNSS_BLACKOUT_MAX"]
        self.sens_gnss_blackout_interval_min = sens_params["SENS_GNSS_BLACKOUT_INTERVAL_MIN"]
        self.sens_gnss_blackout_interval_max = sens_params["SENS_GNSS_BLACKOUT_INTERVAL_MAX"]
        self.sens_gnss_error = sens_params["SENS_GNSS_ERROR"]
        self.sens_gnss_bias = sens_params["SENS_GNSS_BIAS"]
        self.sens_imu_blackout_on = sens_params["SENS_IMU_BLACKOUT_ON"]
        self.sens_imu_blackout_min = sens_params["SENS_IMU_BLACKOUT_MIN"]
        self.sens_imu_blackout_max = sens_params["SENS_IMU_BLACKOUT_MAX"]
        self.sens_imu_blackout_interval_min = sens_params["SENS_IMU_BLACKOUT_INTERVAL_MIN"]
        self.sens_imu_blackout_interval_max = sens_params["SENS_IMU_BLACKOUT_INTERVAL_MAX"]
        self.sens_imu_accel_error = sens_params["SENS_IMU_ACCEL_ERROR"]
        self.sens_imu_gyro_error = sens_params["SENS_IMU_GYRO_ERROR"]
        self.sens_imu_gyro_bias = sens_params["SENS_IMU_GYRO_BIAS"]
        self.sens_imu_sampling = sens_params["SENS_IMU_SAMPLING"]
        self.sens_rgb_blackout = sens_params["SENS_RGB_BLACKOUT"]
        self.sens_rgb_blackout_interval = sens_params["SENS_RGB_BLACKOUT_INTERVAL"]
        self.sens_rgb_sampling = sens_params["SENS_RGB_SAMPLING"]
        self.sens_rgb_stack_size = sens_params["SENS_RGB_STACK_SIZE"]
        self.sens_lidar_blackout = sens_params["SENS_LIDAR_BLACKOUT"]
        self.sens_lidar_top_view = sens_params["SENS_LIDAR_TOP_VIEW"]
        self.sens_lidar_preview = sens_params["SENS_LIDAR_PREVIEW"]
        self.sens_lidar_show_factor = sens_params["SENS_LIDAR_SHOW_FACTOR"]
        self.sens_lidar_channels = sens_params["SENS_LIDAR_CHANNELS"]
        self.sens_lidar_blackout_interval = sens_params["SENS_LIDAR_BLACKOUT_INTERVAL"]
        self.sens_lidar_num_points = sens_params["SENS_LIDAR_NUM_POINTS"]
        self.sens_lidar_frequency = sens_params["SENS_LIDAR_FREQUENCY"]
        self.sens_lidar_range = sens_params["SENS_LIDAR_RANGE"]
        self.sens_lidar_sampling = sens_params["SENS_LIDAR_SAMPLING"]
        self.sens_spd_sas_blackout_on = sens_params["SENS_SPD_SAS_BLACKOUT_ON"]
        self.sens_spd_sas_blackout_min = sens_params["SENS_SPD_SAS_BLACKOUT_MIN"]
        self.sens_spd_sas_blackout_max = sens_params["SENS_SPD_SAS_BLACKOUT_MAX"]
        self.sens_spd_sas_blackout_interval_min = sens_params["SENS_SPD_SAS_BLACKOUT_INTERVAL_MIN"]
        self.sens_spd_sas_blackout_interval_max = sens_params["SENS_SPD_SAS_BLACKOUT_INTERVAL_MAX"]
        self.sens_spd_sas_sampling = sens_params["SENS_SPD_SAS_SAMPLING"]
        self.sens_spd_sas_error = sens_params["SENS_SPD_SAS_ERROR"]
        self.label_colors = sens_params["LABEL_COLORS"]

        # INICIANDO AMBIENTE DE SIMULAÇÃO
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()

        # DEFINE COMANDOS BASE PARA FAZER O SPAWN DOS ATORES DA SIMULAÇÃO
        self.spawn_actor = carla.command.SpawnActor
        #self.set_autopilot = carla.command.SetAutopilot
        # self.future_actor = carla.command.FutureActor
        self.destroy_actor = carla.command.DestroyActor

        # DEFINE VARIÁVEIS DE IDENTIFICAÇÃO REFERENTES AOS OBJETOS QUE SERÃO CRIADOS
        self.walkers_list = []
        self.all_id = []
        self.veh_spawn_points = []
        self.walker_spawn_points = []
        self.all_actors = []
        self.all_sensors = []
        self.vehicles_list = []
        self.props_list = []
        self.ego_vehicle = []
        self.npc_vehicle = []
        self.static_props = []
        self.pedestrians = []
        self.gt_input = []
        self.gt_obj_num = 0
        self.new_episode = False
        self.horizonte_atual = 0
        self.training_atual = 0
        self.episodio_atual = 0
        self.reward_atual = 0
        self.best_reward = 0
        self.reward_inst = 0
        self.simulation_reset = False
        self.sim_total_time = 0
        self.sim_last_total_time = 0
        self.eval = False  # Indica para o módulo top-view se está ocorrendo a fase de evaluation
        self.ignored_spawn_number = 0

        # DEFINE VARIÁVEL QUE IRÁ CONTROLAR A SIMULAÇÃO
        self.simulation_status = None

        # RANDOM / GRADUAL RANDOM
        self.init_gradual_random_ep = self.gradual_random_init_ep_change
        self.current_gradual_random_ep = self.gradual_random_init_ep_change#+1
        self.gradual_random_run_once = True
        self.chosen_random_map = ""

        self.load_world()  # Carrega o mapa e clima

        # ========= CONFIGURA REDE YOLO PARA DETECÇÃO DE OBJETOS POR IMAGENS ===========
        # SETANDO OS PARÂMETROS DA REDE NEURAL YOLO
        if self.rgb_mode == "YOLO":
            net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            # SETANDO OS PARÂMETROS DA REDE NEURAL
            self.yolo_model = cv2.dnn_DetectionModel(net)
            self.yolo_model.setInputParams(size=(416, 416), scale=1 / 255)
            # CARREGA AS CLASSES
            self.yolo_class_names = []
            with open("coco.names", "r") as f:
                self.yolo_class_names = [cname.strip() for cname in f.readlines()]

        # Initialize deque with zero-images one array for each image
        if self.sens_rgb:
            self.stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(self.sens_rgb_stack_size)], maxlen=4)

        #simulação de blackout de sensores (falha simulada)
        if self.sensors_blackout:
            if self.sens_gnss_blackout_on:
                self.sensors_blackout_status_gnss = False
                self.sensors_blackout_time_gnss = 0
                self.sensors_blackout_interval_gnss = 0
                self.sensors_blackout_interval_gnss_actual = random.randrange(self.sens_gnss_blackout_interval_min, self.sens_gnss_blackout_interval_max)
                self.sensors_blackout_time_gnss_actual = 0
            if self.sens_imu_blackout_on:
                self.sensors_blackout_status_imu = False
                self.sensors_blackout_time_imu = 0
                self.sensors_blackout_interval_imu = 0
                self.sensors_blackout_interval_imu_actual = random.randrange(self.sens_imu_blackout_interval_min,
                                                                              self.sens_imu_blackout_interval_max)
            if self.sens_rgb_blackout > 0:
                self.sensors_blackout_status_rgb = False
                self.sensors_blackout_time_rgb = 0
                self.sensors_blackout_interval_rgb = 0
            if self.sens_lidar_blackout > 0:
                self.sensors_blackout_status_lidar = False
                self.sensors_blackout_time_lidar = 0
                self.sensors_blackout_interval_lidar = 0
            if self.sens_spd_sas_blackout_on:
                self.sensors_blackout_status_spd_sas = False
                self.sensors_blackout_time_spd_sas = 0
                self.sensors_blackout_interval_spd_sas = 0
                self.sensors_blackout_interval_spd_sas_actual = random.randrange(self.sens_spd_sas_blackout_interval_min,
                                                                              self.sens_spd_sas_blackout_interval_max)

    # ======= CARREGA O MUNDO E CONFIGURA A SIMULAÇÃO =======
    def load_world(self):
        if self.map == "Random":  # and self.simulation_reset == False:
            random_map = random.choice(self.random_maps)  # garante que o novo mapa é diferente
            while random_map == self.chosen_random_map:
                random_map = random.choice(self.random_maps)
            chosen_map = random_map
            self.client.load_world(chosen_map, True)
            self.chosen_random_map = chosen_map
        elif self.map == "Gradual_Random":  # and self.simulation_reset == True:
            if self.gradual_random_run_once:  # carrega o mapa apenas na primeira vez
                random_map = random.choice(self.random_maps)  # garante que o novo mapa é diferente
                while random_map == self.chosen_random_map:
                    random_map = random.choice(self.random_maps)
                chosen_map = random_map
                self.client.load_world(chosen_map, True)
                self.chosen_random_map = chosen_map
                self.gradual_random_run_once = False
            if self.current_gradual_random_ep <= 0:  # rodou o número de episódios atual no mesmo mapa
                random_map = random.choice(self.random_maps)  # garante que o novo mapa é diferente
                while random_map == self.chosen_random_map:
                    random_map = random.choice(self.random_maps)
                chosen_map = random_map
                self.client.load_world(chosen_map, True)
                self.chosen_random_map = chosen_map
                self.current_gradual_random_ep = self.current_gradual_random_ep - 1
                print(self.current_gradual_random_ep)
                if self.init_gradual_random_ep-self.gradual_random_rate > 0:  # Decrementa o num. de episódios necessário para trocar de mapa
                    self.current_gradual_random_ep = self.init_gradual_random_ep-self.gradual_random_rate
                    self.init_gradual_random_ep = self.init_gradual_random_ep-self.gradual_random_rate
                else:
                    self.current_gradual_random_ep = 1  # Trocar de mapa sempre
        else:
            self.client.load_world(self.map, True)

        '''# Verifica se o mapa já foi carregado, para economizar tempo
        if map.name.split("/")[2] == MAP:
            pass
        else:
            self.client.load_world(MAP, True)
            self.world = self.client.get_world()
        '''

        self.world.wait_for_tick()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.vehicle_distance)
        #self.traffic_manager.global_percentage_speed_difference(VEHICLE_SPEED_DIFF)
        self.color_converter = carla.ColorConverter

        # MUDA CONDIÇÕES CLIMÁTICAS SIMULADAS
        if self.custom_weather:  # Clima conforme parâmetros setados
            weather = self.world.get_weather()
            weather.sun_altitude_angle = self.sun_altitude
            weather.precipitation = self.precipitation_value
            weather.precipitation_deposits = self.precipitation_deposits
            weather.cloudiness = self.cloudiness
            weather.fog_density = self.fog_density
            weather.fog_distance = self.fog_distance
            self.world.set_weather(weather)
        else:  # Habilita um dos presets
            self.weather_presets = self.find_weather_presets()
            #preset = self.weather_presets[WEATHER_PRESET]
            self.world.set_weather(self.weather_presets[self.weather_preset][0])

            # DEFINE A LISTA DE SPAWN POINTS E ATORES NA SIMULAÇÃO - EXECUÇÃO INICIAL
            self.veh_spawn_points = self.world.get_map().get_spawn_points()
            # self.all_actors = self.world.get_actors(self.all_id)

            # CHECA SE EXISTEM SPAWN POINTS SUFICIENTES NO MAPA PARA CARROS
            spawn_points_number = len(self.veh_spawn_points)
            if self.ego_vehicle_num + self.npc_vehicle_num + self.static_props_num <= spawn_points_number:
                random.shuffle(self.veh_spawn_points)
            else:
                print("Foram pedidos %d veículos, mas foram encontrados apenas %d spawn points" % (
                    self.ego_vehicle_num + self.npc_vehicle_num + self.static_props_num, spawn_points_number))
                raise SystemExit(0)

        # ======= FUNÇÃO AUXILIAR PARA UTILIZAÇÃO DE CLIMAS PRÉ-DEFINIDOS =======
    def find_weather_presets(self):
        """Method to find weather presets"""
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

        def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    # ===== DESTRÓI VEÍCULOS E PEDESTRES PARA REINICIAR SIMULAÇÃO =====
    def reset(self):

        # ELIMINA OS PEDESTRES DA SIMULAÇÃO
        # mostra dados em formato de log
        print('\nDestruído(s) %d pedestre(s)' % len(self.walkers_list))

        batch = []
        for pedestrian in self.all_actors:
            if "walker.pedestrian" in pedestrian.type_id:
                batch.append(self.destroy_actor(pedestrian.id))
        self.client.apply_batch(batch)

        # PARA A MEDIÇÃO E ELIMINA OS SENSORES DE TODOS OS CARROS
        if self.vehicles_list is not None:
            for sensor in self.all_sensors:
                sensor.stop()
                sensor.destroy()

        if self.sens_spd_sas:
            for veh in self.ego_vehicle:
                veh.sens_spd_sas.stop()

        # ELIMINA OS VEÍCULOS DA SIMULAÇÃO
        time.sleep(0.5)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        # mostra dados em formato de log
        print('Destruído(s) %d veículo(s) EGO' % self.ego_vehicle_num)
        print('Destruído(s) %d veículo(s) NPC' % self.npc_vehicle_num)

        # ELIMINA OS OBSTÁCULOS DA SIMULAÇÃO
        time.sleep(0.5)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.props_list])
        # mostra dados em formato de log
        print('Destruído(s) %d obstáculo(s)' % len(self.props_list))

        # DEFINE A LISTA DE SPAWN POINTS E ATORES NA SIMULAÇÃO - EXECUÇÃO A CADA NOVO EPISÓDIO
        self.walkers_list = []
        self.all_id = []
        self.walker_spawn_points = []
        self.all_actors = []
        self.all_sensors = []
        self.vehicles_list = []
        self.ego_vehicle = []
        self.props_list = []
        #self.simulation_reset = False

        # LIMPA O STACK DE IMAGENS
        if self.sens_rgb:
            self.stacked_frames = deque([np.zeros((84, 84), dtype=np.int32) for i in range(self.sens_rgb_stack_size)], maxlen=4)

        # REDEFINE OS SPAWN-POINTS E RECARREGA MUNDO SIMULADO
        #random.shuffle(self.veh_spawn_points)
        self.load_world()  # carrega o mundo simulado
        # self.all_actors = self.world.get_actors(self.all_id)

        # Change weather conditions
        #if CUSTOM_WEATHER:  # Clima conforme parâmetros setados
        #    weather = self.world.get_weather()
        #    weather.sun_altitude_angle = SUN_ALTITUDE
        #    weather.precipitation = PRECIPITATION_VALUE
        #    weather.precipitation_deposits = PRECIPITATION_DEPOSITS
        #    weather.cloudiness = CLOUDINESS
        #    weather.fog_density = FOG_DENSITY
        #    weather.fog_distance = FOG_DISTANCE
        #    self.world.set_weather(weather)
        #else:  # Habilita um dos presets
        #    self.weather_presets = self.find_weather_presets()
            #preset = self.weather_presets[WEATHER_PRESET]
        #    self.world.set_weather(self.weather_presets[WEATHER_PRESET][0])

    # ===== DEFINIÇÃO DAS FUNÇÕES DE CALLBACK COM A LEITURA DOS SENSORES (ARMAZENA APENAS ÚLTIMOS VALORES) =====
    def gnss_callback(self, gnss, veh_num):

        self.ego_vehicle[veh_num].sens_gnss = gnss
        self.ego_vehicle[veh_num].sens_gnss_preview = self.sens_gnss_preview

        # CONVERSÃO DA LEITURA DO GNSS P/ POSIÇÕES DENTRO DO CARLA
        a = 6378137
        b = 6356752.3142
        f = (a - b) / a
        e_sq = f * (2 - f)

        def geodetic_to_ecef(lat, lon, h):
            # (lat, lon) in WSG-84 degrees
            # h in meters
            lamb = math.radians(lat)
            phi = math.radians(lon)
            s = math.sin(lamb)
            N = a / math.sqrt(1 - e_sq * s * s)

            sin_lambda = math.sin(lamb)
            cos_lambda = math.cos(lamb)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            x = (h + N) * cos_lambda * cos_phi
            y = (h + N) * cos_lambda * sin_phi
            z = (h + (1 - e_sq) * N) * sin_lambda

            return x, y, z

        def ecef_to_enu(x, y, z, lat0, lon0, h0):
            lamb = math.radians(lat0)
            phi = math.radians(lon0)
            s = math.sin(lamb)
            N = a / math.sqrt(1 - e_sq * s * s)

            sin_lambda = math.sin(lamb)
            cos_lambda = math.cos(lamb)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            x0 = (h0 + N) * cos_lambda * cos_phi
            y0 = (h0 + N) * cos_lambda * sin_phi
            z0 = (h0 + (1 - e_sq) * N) * sin_lambda

            xd = x - x0
            yd = y - y0
            zd = z - z0

            xEast = -sin_phi * xd + cos_phi * yd
            yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
            zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

            return xEast, yNorth, zUp

        def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
            x, y, z = geodetic_to_ecef(lat, lon, h)

            return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

        # VALORES DE ORIGEM ENCONTRADOS NO ARQUIVO TOWN02.XODR (FORMATO OPENDRIVE)
        origin_latitude = 0.0  # 49.000000
        origin_longitude = 0.0  # 8.00000
        origin_altitude = 0.0  # 0.0000
        # print("Veículo ", veh_num, " GNSS: ", str(gnss))  # + '\n')

        # LÊ OS VALORES RECEBIDOS PELO SENSOR GNSS
        point_latitude = gnss.latitude
        point_longitude = gnss.longitude
        point_altitude = gnss.altitude

        x, y, z = geodetic_to_ecef(point_latitude, point_longitude, point_altitude)
        carla_x, carla_y, carla_z = ecef_to_enu(x, y, z, origin_latitude, origin_longitude, origin_altitude)

        self.ego_vehicle[veh_num].sens_gnss.location = carla.Location(x=carla_x, y=-carla_y, z=carla_z)

        # Simula blackout dos sensores, caso selecionado. Durante blackout, valor anterior do sensor é mantido.
        if self.sensors_blackout and self.sens_gnss_blackout_on:
            if self.sensors_blackout_status_gnss == False:
                self.ego_vehicle[veh_num].last_value_gnss = self.ego_vehicle[veh_num].sens_gnss.location
                if time.time()-self.sensors_blackout_interval_gnss > self.sensors_blackout_interval_gnss_actual:
                    self.sensors_blackout_status_gnss = True
                    self.sensors_blackout_time_gnss_actual = random.randrange(self.sens_gnss_blackout_min,
                                                                                self.sens_gnss_blackout_max)
                    self.sensors_blackout_time_gnss = time.time()

                    #print("Blackout ON")
            else:  # True
                self.ego_vehicle[veh_num].sens_gnss.location = self.ego_vehicle[veh_num].last_value_gnss
                if time.time() - self.sensors_blackout_time_gnss > self.sensors_blackout_time_gnss_actual:
                    self.sensors_blackout_status_gnss = False
                    self.sensors_blackout_interval_gnss_actual = random.randrange(self.sens_gnss_blackout_interval_min,
                                                                                  self.sens_gnss_blackout_interval_max)
                    self.sensors_blackout_interval_gnss = time.time()
                    #print("Blackout OFF")

    def imu_callback(self, imu, veh_num):

        self.ego_vehicle[veh_num].sens_imu = imu

        # CÁLCULO PARA FACILITAR A LEITURA DOS DADOS DO IMU
        limits = (-99.9, 99.9)
        self.ego_vehicle[veh_num].sens_imu.ue_accelerometer = (
            max(limits[0], min(limits[1], imu.accelerometer.x)),
            max(limits[0], min(limits[1], imu.accelerometer.y)),
            max(limits[0], min(limits[1], imu.accelerometer.z)))
        self.ego_vehicle[veh_num].sens_imu.ue_gyroscope = (
            max(limits[0], min(limits[1], math.degrees(imu.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(imu.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(imu.gyroscope.z))))

        # CALCULA GRAUS E DIREÇÃO
        compass = math.degrees(imu.compass)
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''

        self.ego_vehicle[veh_num].sens_imu.ue_compass_degrees = compass
        self.ego_vehicle[veh_num].sens_imu.ue_compass_heading = heading

        # Simula blackout dos sensores, caso selecionado. Durante blackout, valor anterior do sensor é mantido.
        if self.sensors_blackout and self.sens_imu_blackout_on:
            if self.sensors_blackout_status_imu == False:
                self.ego_vehicle[veh_num].last_value_imu_ue_accelerometer = self.ego_vehicle[veh_num].sens_imu.ue_accelerometer
                self.ego_vehicle[veh_num].last_value_imu_ue_gyroscope = self.ego_vehicle[veh_num].sens_imu.ue_gyroscope
                self.ego_vehicle[veh_num].last_value_imu_ue_compass_degrees = self.ego_vehicle[veh_num].sens_imu.ue_compass_degrees
                if time.time()-self.sensors_blackout_interval_imu > self.sensors_blackout_interval_imu_actual:
                    self.sensors_blackout_status_imu = True
                    self.sensors_blackout_time_imu_actual = random.randrange(self.sens_imu_blackout_min,
                                                                              self.sens_imu_blackout_max)
                    self.sensors_blackout_time_imu = time.time()
                    #print("Blackout ON")
            else:  # True
                self.ego_vehicle[veh_num].sens_imu.ue_accelerometer = self.ego_vehicle[veh_num].last_value_imu_ue_accelerometer
                self.ego_vehicle[veh_num].sens_imu.ue_gyroscope = self.ego_vehicle[veh_num].last_value_imu_ue_gyroscope
                self.ego_vehicle[veh_num].sens_imu.ue_compass_degrees = self.ego_vehicle[veh_num].last_value_imu_ue_compass_degrees
                if time.time()-self.sensors_blackout_time_imu > self.sensors_blackout_time_imu_actual:
                    self.sensors_blackout_status_imu = False
                    self.sensors_blackout_time_imu_actual = random.randrange(self.sens_imu_blackout_interval_min,
                                                                              self.sens_imu_blackout_interval_max)
                    self.sensors_blackout_interval_imu = time.time()
                    #print("Blackout OFF")

        # CRIA VARIÁVEL QUE IRÁ ALIMENTAR A RN
        #self.ego_vehicle[veh_num].sens_imu_input = [compass , heading]

    def obs_callback(self, obs, veh_num):

        self.ego_vehicle[veh_num].sens_obs = obs
        # print("Veículo ", veh_num, " OBS: ", str(obs))  # print("Obstacle detected:\n" + str(obs) + '\n')

    def col_callback(self, col, veh_num):

        self.ego_vehicle[veh_num].sens_col = col
        print("Colisão:", col)
        print("Velocidade", self.ego_vehicle[veh_num].sens_spd_sas_speed)
        if round(self.ego_vehicle[veh_num].sens_spd_sas_speed) == 0 and self.simulation_reset == False:  # reinicia o episódio se algum carro colidir e parar
            self.simulation_reset = True
            print("Veículo ", veh_num, " colidiu, reiniciando episódio.")

    def rgb_callback(self, rgb, veh_num):

        self.ego_vehicle[veh_num].sens_rgb = rgb

        # REALIZA O PRÉ-PROCESSAMENTO DA IMAGEM
        def preprocess_frame(frame):
            # Greyscale frame not necessary
            # x = np.mean(frame,-1)

            # Crop the screen (remove the roof because it contains no information)
            # [Up: Down, Left: right]
            # cropped_frame = frame[80:, :]

            # Normalize Pixel Values
            normalized_frame = frame / 255.0

            # Resize
            preprocessed_frame = transform.resize(normalized_frame, [84, 84])

            return preprocessed_frame

        # MONTA UM STACK DE X IMAGENS QUE IRÁ ALIMENTAR A RN
        def stack_frames(simulation, stacked_frames, state, is_new_episode,sens_rgb_stack_size):
            # Preprocess frame
            frame = preprocess_frame(state)

            if is_new_episode:
                # Clear our stacked_frames
                stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(sens_rgb_stack_size)], maxlen=4)

                # Because we're in a new episode, copy the same frame 4x
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)
                stacked_frames.append(frame)

                # Stack the frames
                stacked_state = np.stack(stacked_frames, axis=2)

                simulation.new_episode = True

            else:
                # Append frame to deque, automatically removes the oldest frame
                stacked_frames.append(frame)

                # Build the stacked state (first dimension specifies different frames)
                stacked_state = np.stack(stacked_frames, axis=2)

            return stacked_state, stacked_frames

        # SALVA AS IMAGENS NO HD:
        # cv2.imwrite('tutorial/output/%.6d.png' % rgb.frame, image)
        # rgb.save_to_disk('tutorial/output/Sem_%.6d.jpg' % rgb.frame, carla.ColorConverter.CityScapesPalette)


        if self.rgb_mode == "BINARY":  # MÉTODO VIA BINARIZAÇÃO (SEMÂNTICO)
            image = np.array(rgb.raw_data)
            image = image.reshape((self.im_height, self.im_width, 4))
            image = image[:, :, :3]
            semantic = image[:, :, 2]  # extrai apenas o layer que possui os dados semânticos
            pedestres = semantic == 4  # pedestres
            veiculos = semantic == 10  # veículos
            if self.static_props_num > 0:
                props = semantic == 20  # cones
                image = np.array(255 * pedestres + 255 * veiculos + 255 * props).astype(np.uint8)
                self.ego_vehicle[veh_num].sens_rgb_props = np.array(255 * props).astype(
                    np.uint8)  # apenas props
            else:
                image = np.array(255 * pedestres + 255 * veiculos).astype(np.uint8)
            self.ego_vehicle[veh_num].sens_rgb_data = image  # imagem completa
            self.ego_vehicle[veh_num].sens_rgb_pedestres = np.array(255 * pedestres).astype(np.uint8)  # apenas pedestres
            self.ego_vehicle[veh_num].sens_rgb_veiculos = np.array(255 * veiculos).astype(np.uint8)  # apenas veículos

        elif self.rgb_mode == "SEMANTIC":  # RGB SEMÂNTICO PURO
            rgb.convert(self.color_converter.CityScapesPalette)
            image = np.array(rgb.raw_data)
            image = image.reshape((self.im_height, self.im_width, 4))
            image = image[:, :, :3]
            image = np.array(image)  # esse passo é necessário para que o comando image_show funcione
            self.ego_vehicle[veh_num].sens_rgb_data = image

        # ================== CÓDIGO PARA DETECÇÃO DE VÁRIOS OBJETOS - YOLO (RGB) ==================
        elif self.rgb_mode == "YOLO":  # executa apenas a cada X execuções ou sempre se o valor for 999
            image = np.array(rgb.raw_data)
            image = image.reshape((self.im_height, self.im_width, 4))
            image = image[:, :, :3]
            image = np.array(image)
            classes, scores, boxes = self.yolo_model.detect(image, 0.1, 0.2)

            try:
                for (classid, score, box) in zip(classes, scores, boxes):
                    color = self.yolo_colors[int(classid) % len(self.yolo_colors)]
                    label = f"{self.yolo_class_names[classid[0]]} : {score}"
                    cv2.rectangle(image, box, color, 2)
                    cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    self.ego_vehicle[veh_num].sens_rgb_objid = self.yolo_class_names[classid[0]]
            except:
                pass
            self.ego_vehicle[veh_num].sens_rgb_data = image

        # FAZ BUFFER DE X IMAGENS
        state, stacked_frames = stack_frames(self,self.stacked_frames, image, self.new_episode, self.sens_rgb_stack_size)
        self.ego_vehicle[veh_num].sens_rgb_input = stacked_frames

        # Simula blackout dos sensores, caso selecionado. Durante blackout, valor anterior do sensor é mantido.
        if self.sensors_blackout and self.sens_rgb_blackout >0:
            if self.sensors_blackout_status_rgb == False:
                self.ego_vehicle[veh_num].last_value_rgb = self.ego_vehicle[veh_num].sens_rgb_input
                if time.time()-self.sensors_blackout_interval_rgb > self.sens_rgb_blackout_interval:
                    self.sensors_blackout_status_rgb = True
                    self.sensors_blackout_time_rgb = time.time()
                    #print("Blackout ON")
            else:  # True
                self.ego_vehicle[veh_num].sens_rgb_input = self.ego_vehicle[veh_num].last_value_rgb
                if time.time() - self.sensors_blackout_time_rgb > self.sens_rgb_blackout:
                    self.sensors_blackout_status_rgb = False
                    self.sensors_blackout_interval_rgb = time.time()
                    #print("Blackout OFF")


        # ================== CÓDIGO PARA DETECÇÃO DE APENAS UM OBJETO - YOLO =====================
        #classes, scores, boxes = self.yolo_model.detect(image, 0.1, 0.2)
        # print(scores)
        #self.ego_vehicle[veh_num].sens_rgb_objid = ""
        #try:
        #    index = np.argmax(scores)
        #    color = YOLO_COLORS[int(classes[index]) % len(YOLO_COLORS)]
        #    label = f"{self.yolo_class_names[classes[index][0]]} : {scores[index]}"
        #    print(label)
        #    cv2.rectangle(image, boxes[index], color, 2)
        #    cv2.putText(image, label, (boxes[index][0], boxes[index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
        #                2)
        #    self.ego_vehicle[veh_num].sens_rgb_objid = str(self.yolo_class_names[classes[index][0]])
        #except:
        #    pass
        #self.ego_vehicle[veh_num].sens_rgb_data = image

        # ================= MÉTODO VIA DETECÇÃO DE CONTORNOS (SEMÂNTICO) =====================
        # contours, hierarchy = cv2.findContours(mask,
        #                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # for c in contours:
        #    M = cv2.moments(c)
        #    cX = int(M["m10"]/M["m00"])
        #    cY = int(M["m01"]/M["m00"])
        #    cv2.circle(image,(cX,cY), 7, (255,255,255), -1)
        # cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)
        # print(cv2.countNonZero(mask))

    def lidar_callback(self, lidar, veh_num, point_list):  # (self, lidar, veh_num):

        self.ego_vehicle[veh_num].sens_lidar = lidar
        self.ego_vehicle[veh_num].sens_lidar_top_view = self.sens_lidar_top_view
        self.ego_vehicle[veh_num].sens_lidar_preview = self.sens_lidar_preview

        # lidar.save_to_disk('~/tutorial/new_lidar_output/%.6d.ply' % lidar.raw_data)

        data = np.frombuffer(lidar.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system

        data = data[::self.sens_lidar_show_factor]
        points = np.array([data['x'], data['y'], data['z']]).T  #Antes estava -data['y']

        #savetxt('tutorial/lidar/%.6d.csv' % lidar.frame, points, delimiter=';')

        # # An example of adding some noise to our data if needed:
        # points += np.random.uniform(-0.05, 0.05, size=points.shape)

        # Colorize the pointcloud based on the CityScapes color palette
        labels = np.array(data['ObjTag'])

        colors = self.label_colors[labels]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(colors)

        self.ego_vehicle[veh_num].sens_lidar_points = point_list.points
        self.ego_vehicle[veh_num].sens_lidar_color = point_list.colors

        # Simula blackout dos sensores, caso selecionado. Durante blackout, valor anterior do sensor é mantido.
        if self.sensors_blackout and self.sens_lidar_blackout>0:
            if self.sensors_blackout_status_lidar == False:
                self.ego_vehicle[veh_num].last_value_lidar_points = self.ego_vehicle[veh_num].sens_lidar_points
                self.ego_vehicle[veh_num].last_value_lidar_color = self.ego_vehicle[veh_num].sens_lidar_color
                if time.time()-self.sensors_blackout_interval_lidar > self.sens_lidar_blackout_interval:
                    self.sensors_blackout_status_lidar = True
                    self.sensors_blackout_time_lidar = time.time()
                    #print("Blackout ON")
            else:  # True
                self.ego_vehicle[veh_num].sens_lidar_points = self.ego_vehicle[veh_num].last_value_lidar_points
                self.ego_vehicle[veh_num].sens_lidar_color = self.ego_vehicle[veh_num].last_value_lidar_color
                if time.time() - self.sensors_blackout_time_lidar > self.sens_lidar_blackout:
                    self.sensors_blackout_status_lidar = False
                    self.sensors_blackout_interval_lidar = time.time()
                    #print("Blackout OFF")


    def speed_sas_callback(self, veh_num):

        # CÁLCULO DA VELOCIDADE
        speed = self.ego_vehicle[veh_num].get_velocity()
        speed = 3.6 * math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)
        self.ego_vehicle[veh_num].sens_spd_sas_speed = speed

        # ÂNGULO DO VOLANTE
        #sas = self.ego_vehicle[veh_num].get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        angle = self.ego_vehicle[veh_num].agent._local_planner._vehicle_controller.past_steering
        angle = angle*360  # Exibe ângulo em graus
        self.ego_vehicle[veh_num].sens_spd_sas_angle = angle

        # ERRO PADRÃO, SE VALOR > 0
        if self.sens_spd_sas_error > 0:
            self.ego_vehicle[veh_num].sens_spd_sas_speed = np.random.normal(loc=speed, scale=self.sens_spd_sas_error, size=None)
            self.ego_vehicle[veh_num].sens_spd_sas_angle = np.random.normal(loc=angle, scale=self.sens_spd_sas_error, size=None)

        # Simula blackout dos sensores, caso selecionado. Durante blackout, valor anterior do sensor é mantido.
        if self.sensors_blackout and self.sens_spd_sas_blackout_on:
            if self.sensors_blackout_status_spd_sas == False:
                self.ego_vehicle[veh_num].last_value_spd_sas_speed = self.ego_vehicle[veh_num].sens_spd_sas_speed
                self.ego_vehicle[veh_num].last_value_spd_sas_angle = self.ego_vehicle[veh_num].sens_spd_sas_angle
                if time.time()-self.sensors_blackout_interval_spd_sas > self.sensors_blackout_interval_spd_sas_actual:
                    self.sensors_blackout_status_spd_sas = True
                    self.sensors_blackout_time_spd_sas_actual = random.randrange(self.sens_spd_sas_blackout_min,
                                                                             self.sens_spd_sas_blackout_max)
                    self.sensors_blackout_time_spd_sas = time.time()
                    #print("Blackout ON")
            else:  # True
                self.ego_vehicle[veh_num].sens_spd_sas_speed = self.ego_vehicle[veh_num].last_value_spd_sas_speed
                self.ego_vehicle[veh_num].sens_spd_sas_angle = self.ego_vehicle[veh_num].last_value_spd_sas_angle
                if time.time() - self.sensors_blackout_time_spd_sas > self.sensors_blackout_time_spd_sas_actual:
                    self.sensors_blackout_status_spd_sas = False
                    self.sensors_blackout_interval_spd_sas_actual = random.randrange(self.sens_spd_sas_blackout_interval_min,
                                                                                  self.sens_spd_sas_blackout_interval_max)
                    self.sensors_blackout_interval_spd_sas = time.time()
                    #print("Blackout OFF")


    # ===== CARREGA OS PEDESTRES NA SIMULAÇÃO =====
    def spawn_pedestrians(self):  # PENSAR EM COMO FAZER ESSA FUNÇÃO DAR SPAWN APENAS DE UM PEDESTRE
        # 1. take random location and spawn walker object
        # self.batch = []
        walker_speed = []

        # TENTA CARREGAR PEDESTRES ATÉ O NÚMERO REQUISITADO SER ATINGIDO
        while len(self.walkers_list) < self.pedestrian_num:
            try:
                # Spawn Walkers
                bp_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    self.walker_spawn_points.append(spawn_point)
                walker_bp = random.choice(bp_walkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # tenta fazer o spawn, se der erro, o restante do código não é executado
                result = self.world.spawn_actor(walker_bp, spawn_point)
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if random.random() > self.percentage_pedestrians_running:
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                self.pedestrians.append(result)
                self.walkers_list.append({"id": result.id})
                # Muda a visão para o local onde aconteceu o spawn - APENAS DEBUG
                spectator = self.world.get_spectator()
                spectator.set_transform(result.get_transform())
            except:
                pass
                # print("Colisão durante Spawn") - APENAS DEBUG

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(self.spawn_actor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.wait_for_tick()  # Para modo síncrono, usar world.tick()
        time.sleep(0.1)

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(self.percentage_pedestrians_crossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        # mostra dados em formato de log
        print('Criado(s) %d pedestres' % (len(self.walkers_list)))

    # ===== CARREGA CARRO COM OU SEM SENSORES =====
    def spawn_vehicle(self, veh_num, spawn_ego):
        # CARREGA BLUEPRINTS DE CARROS E SPAWN POINTS
        bp_vehicle = self.world.get_blueprint_library().filter("vehicle.*")
        bp_vehicle = [x for x in bp_vehicle if int(x.get_attribute('number_of_wheels')) == 4]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('isetta')]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('carlacola')]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('cybertruck')]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('firetruck')]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('t2')]
        bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('ambulance')]
        bp_vehicle = sorted(bp_vehicle, key=lambda bp: bp.id)
        bp_vehicle = random.choice(bp_vehicle)
        if spawn_ego:  # CRIA CARRO COMO EGO
            if self.centralized_spawn and self.chosen_random_map == "Town02":  # Força o spawn na região central do mapa (por enquanto apenas na Town02)
                while not ((43 < self.veh_spawn_points[veh_num + self.ignored_spawn_number].location.x < 138) and (187 < self.veh_spawn_points[veh_num + self.ignored_spawn_number].location.y < 237)):
                    self.ignored_spawn_number += 1
                    if self.ignored_spawn_number > len(self.veh_spawn_points):
                        print("Não existem spawn points suficientes no centro do mapa. Desativar Centralized Spawn!")
                        raise SystemExit(0)
                transform = self.veh_spawn_points[
                        veh_num + self.ignored_spawn_number]  # segue sequência que os carros forem criados, começando em veh_num[0]
                self.ignored_spawn_number = 0
            else:
                transform = self.veh_spawn_points[veh_num]

        else:  # CRIA CARRO COMO NPC
            transform = self.veh_spawn_points[veh_num+self.ego_vehicle_num]  # pula os spawn points usados pelos EGO

        # CONFIGURA ATRIBUTOS COLOR, DRIVER_ID E ROLE_NAME
        if bp_vehicle.has_attribute('color'):
            color = random.choice(bp_vehicle.get_attribute('color').recommended_values)
            bp_vehicle.set_attribute('color', color)
        if bp_vehicle.has_attribute('driver_id'):
            driver_id = random.choice(bp_vehicle.get_attribute('driver_id').recommended_values)
            bp_vehicle.set_attribute('driver_id', driver_id)

        if spawn_ego:  # CRIA CARRO COMO EGO
            bp_vehicle.set_attribute('role_name', 'EGO ' + str(veh_num + 1))  # EGO 1, EGO 2
            # spawn the cars and set their autopilot
            self.ego_vehicle.append(self.world.spawn_actor(bp_vehicle, transform))
            self.world.wait_for_tick()
            time.sleep(0.1)
            self.vehicles_list.append(self.ego_vehicle[veh_num].id)
            if self.vehicle_agent == "BASIC":
                self.ego_vehicle[veh_num].agent = BasicAgent(self.ego_vehicle[veh_num])
                #spawn_point = random.choice(self.veh_spawn_points)
                #self.ego_vehicle[veh_num].agent.set_destination((spawn_point.location.x,
                                               #spawn_point.location.y,
                                               #spawn_point.location.z))
                if self.vehicle_speed != "Limit":
                    self.ego_vehicle[veh_num].agent.follow_speed_limits(value=False)
                    self.ego_vehicle[veh_num].agent.set_target_speed(self.vehicle_speed)
                else:  # valor fixo de velocidade
                    self.ego_vehicle[veh_num].agent.follow_speed_limits(value=True)
            elif self.vehicle_agent == "BEHAVIOR":
                self.ego_vehicle[veh_num].agent = BehaviorAgent(self.ego_vehicle[veh_num], behavior=self.vehicle_behavior)
                #spawn_point = random.choice(self.veh_spawn_points)
                #if spawn_point.location != self.ego_vehicle[veh_num].agent.vehicle.get_location():
                #    destination = spawn_point.location
                #else:
                #    destination = spawn_point.location
                #destination = spawn_point.location
                #self.ego_vehicle[veh_num].agent.set_destination(self.ego_vehicle[veh_num].agent.vehicle.get_location(), destination, clean=True)
                #self.ego_vehicle[veh_num].agent.set_destination(destination, clean=True)
            elif self.vehicle_agent == "SERVER":
                self.ego_vehicle[veh_num].set_autopilot(True, self.traffic_manager.get_port())
                #self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle[veh_num],50)
            else:
                pass  # veículos parados

        else:  # CRIA CARRO COMO NPC
            bp_vehicle.set_attribute('role_name', 'NPC ' + str(veh_num + 1))  # EGO 1, EGO 2
            # spawn the cars and set their autopilot
            self.npc_vehicle.append(self.world.spawn_actor(bp_vehicle, transform))
            self.world.wait_for_tick()
            time.sleep(0.1)
            self.vehicles_list.append(self.npc_vehicle[veh_num].id)
            if self.vehicle_agent == "BASIC":
                self.npc_vehicle[veh_num].agent = BasicAgent(self.npc_vehicle[veh_num])
                #spawn_point = random.choice(self.veh_spawn_points)
                #self.npc_vehicle[veh_num].agent.set_destination((spawn_point.location.x,
                                               #spawn_point.location.y,
                                               #spawn_point.location.z))
            elif self.vehicle_agent == "BEHAVIOR":
                self.npc_vehicle[veh_num].agent = BehaviorAgent(self.npc_vehicle[veh_num], behavior=self.vehicle_behavior)
                spawn_point = random.choice(self.veh_spawn_points)
                if spawn_point.location != self.npc_vehicle[veh_num].agent.vehicle.get_location():
                    destination = spawn_point.location
                else:
                    destination = spawn_point.location
                self.npc_vehicle[veh_num].agent.set_destination(self.npc_vehicle[veh_num].agent.vehicle.get_location(), destination, clean=True)
            elif self.vehicle_agent == "SERVER":
                self.npc_vehicle[veh_num].set_autopilot(True, self.traffic_manager.get_port())
            else:
                pass  # veículos parados

         # Muda a visão para o local onde aconteceu o spawn - APENAS DEBUG
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)

        if spawn_ego:  # CRIA CARRO COMO EGO

            # INICIALIZA OS SENSORES
            self.ego_vehicle[veh_num].sens_gnss = None
            self.ego_vehicle[veh_num].sens_imu = None
            self.ego_vehicle[veh_num].sens_obs = None
            self.ego_vehicle[veh_num].sens_rgb = None
            self.ego_vehicle[veh_num].sens_rgb_objid = ""
            self.ego_vehicle[veh_num].sens_lidar = None
            self.ego_vehicle[veh_num].sens_lidar_points_rotated = []
            self.ego_vehicle[veh_num].static_props_num = self.static_props_num
            self.ego_vehicle[veh_num].sens_gnss_input = None
            self.ego_vehicle[veh_num].sens_lidar_input = None
            self.ego_vehicle[veh_num].prediction = None
            self.ego_vehicle[veh_num].pred_kalman_x = None
            self.ego_vehicle[veh_num].pred_kalman_y = None
            self.ego_vehicle[veh_num].pred_distance = None
            self.ego_vehicle[veh_num].prediction_preview = self.prediction_preview
            self.ego_vehicle[veh_num].last_value_gnss = []
            self.ego_vehicle[veh_num].last_value_imu_ue_accelerometer = [0,0,0]
            self.ego_vehicle[veh_num].last_value_imu_ue_gyroscope = [0,0,0]
            self.ego_vehicle[veh_num].last_value_imu_ue_compass_degrees = 0
            self.ego_vehicle[veh_num].last_value_rgb = []
            self.ego_vehicle[veh_num].last_value_lidar_points = []
            self.ego_vehicle[veh_num].last_value_lidar_color = []
            self.ego_vehicle[veh_num].last_value_spd_sas_speed = 0
            self.ego_vehicle[veh_num].last_value_spd_sas_angle = 0
            self.ego_vehicle[veh_num].stacked_positions = []

            # CONFIGURAÇÃO DO SPEED / STEERING ANGLE SENSOR (SPD_SAS) - CUSTOM SENSOR
            if self.sens_spd_sas:
                self.ego_vehicle[veh_num].sens_spd_sas = Speed_SAS_Sensor(self.sens_spd_sas_sampling, self.speed_sas_callback, veh_num)

            # CONFIGURAÇÃO DO GNSS
            if self.sens_gnss:
                gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
                gnss_location = carla.Location(0, 0, 0)
                gnss_rotation = carla.Rotation(0, 0, 0)
                gnss_transform = carla.Transform(gnss_location, gnss_rotation)
                gnss_bp.set_attribute("sensor_tick", str(self.sens_gnss_sampling))  # Default 3s
                gnss_bp.set_attribute("noise_lat_stddev", str(self.sens_gnss_error))
                gnss_bp.set_attribute("noise_lon_stddev", str(self.sens_gnss_error))
                gnss_bp.set_attribute("noise_lat_bias", str(self.sens_gnss_bias))
                gnss_bp.set_attribute("noise_lon_bias", str(self.sens_gnss_bias))
                ego_gnss = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.ego_vehicle[veh_num],
                                              attachment_type=carla.AttachmentType.Rigid)
                ego_gnss.listen(lambda gnss: self.gnss_callback(gnss, veh_num))
                self.all_sensors.append(ego_gnss)

            # CONFIGURAÇÃO DO IMU
            if self.sens_imu:
                imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
                imu_location = carla.Location(0, 0, 0)
                imu_rotation = carla.Rotation(0, 0, 0)
                imu_transform = carla.Transform(imu_location, imu_rotation)
                imu_bp.set_attribute("noise_accel_stddev_x", str(self.sens_imu_accel_error))
                imu_bp.set_attribute("noise_accel_stddev_y", str(self.sens_imu_accel_error))
                imu_bp.set_attribute("noise_gyro_stddev_x", str(self.sens_imu_gyro_error))
                imu_bp.set_attribute("noise_gyro_stddev_y", str(self.sens_imu_gyro_error))
                imu_bp.set_attribute("noise_gyro_stddev_z", str(self.sens_imu_gyro_error))
                imu_bp.set_attribute("noise_gyro_bias_x", str(self.sens_imu_gyro_bias))
                imu_bp.set_attribute("noise_gyro_bias_y", str(self.sens_imu_gyro_bias))
                imu_bp.set_attribute("noise_gyro_bias_z", str(self.sens_imu_gyro_bias))
                imu_bp.set_attribute("sensor_tick", str(self.sens_imu_sampling))  # Default 3s
                ego_imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.ego_vehicle[veh_num],
                                             attachment_type=carla.AttachmentType.Rigid)
                ego_imu.listen(lambda imu: self.imu_callback(imu, veh_num))
                self.all_sensors.append(ego_imu)

            # CONFIGURAÇÃO DO OBSTACLE DETECTOR
            if self.sens_obs:
                obs_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
                obs_bp.set_attribute("only_dynamics", str(True))
                obs_location = carla.Location(0, 0, 0)
                obs_rotation = carla.Rotation(0, 0, 0)
                obs_transform = carla.Transform(obs_location, obs_rotation)
                ego_obs = self.world.spawn_actor(obs_bp, obs_transform, attach_to=self.ego_vehicle[veh_num],
                                             attachment_type=carla.AttachmentType.Rigid)
                ego_obs.listen(lambda obs: self.obs_callback(obs, veh_num))
                self.all_sensors.append(ego_obs)

            # CONFIGURAÇÃO DO COLLISION DETECTOR
            if self.sens_col:
                col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
                col_location = carla.Location(0, 0, 0)
                col_rotation = carla.Rotation(0, 0, 0)
                col_transform = carla.Transform(col_location, col_rotation)
                #col_bp.set_attribute("sensor_tick", str(SENS_COL_SAMPLING))
                ego_col = self.world.spawn_actor(col_bp, col_transform, attach_to=self.ego_vehicle[veh_num],
                                             attachment_type=carla.AttachmentType.Rigid)
                ego_col.listen(lambda col: self.col_callback(col, veh_num))
                self.all_sensors.append(ego_col)

            # CONFIGURAÇÃO DA CAMERA RGB
            if self.sens_rgb:
                if self.rgb_mode == "YOLO":
                    cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
                else:
                    cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                cam_bp.set_attribute("image_size_x", str(self.im_width))
                cam_bp.set_attribute("image_size_y", str(self.im_height))
                cam_bp.set_attribute("fov", str(105))
                cam_bp.set_attribute("sensor_tick", str(self.sens_rgb_sampling))  # Default 3s
                cam_location = carla.Location(3, 0, 1)  #(3,0,1)
                cam_rotation = carla.Rotation(0, 0, 0)  # (0, 180, 0)
                cam_transform = carla.Transform(cam_location, cam_rotation)
                ego_rgb = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.ego_vehicle[veh_num],
                                             attachment_type=carla.AttachmentType.Rigid)
                ego_rgb.listen(lambda rgb: self.rgb_callback(rgb, veh_num))
                self.all_sensors.append(ego_rgb)

            # CONFIGURAÇÃO LIDAR SEMÂNTICO
            if self.sens_lidar:
                lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
                lidar_bp.set_attribute('channels', str(self.sens_lidar_channels))  # str(32)
                lidar_bp.set_attribute('points_per_second', str(self.sens_lidar_num_points))  # str(90000))
                lidar_bp.set_attribute('rotation_frequency', str(self.sens_lidar_frequency))  # str(40))
                lidar_bp.set_attribute('range', str(self.sens_lidar_range))
                lidar_bp.set_attribute("sensor_tick", str(self.sens_lidar_sampling))  # Default 0
                lidar_location = carla.Location(0, 0, 2)
                lidar_rotation = carla.Rotation(0, 0, 0)
                lidar_transform = carla.Transform(lidar_location, lidar_rotation)
                ego_lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle[veh_num],
                                               attachment_type=carla.AttachmentType.Rigid)
                point_list = o3d.geometry.PointCloud()
                ego_lidar.listen(lambda lidar: self.lidar_callback(lidar, veh_num, point_list))
                self.all_sensors.append(ego_lidar)

    # ===== CARREGA OBJETOS ESTÁTICOS NOS LUGARES DE SPAWN DE VEÍCULOS =====
    def spawn_props(self,props_num):
        # CARREGA BLUEPRINTS DE OBJETOS E SPAWN POINTS
        bp_props = self.world.get_blueprint_library().filter("static.prop*")
        bp_props = [x for x in bp_props if x.id.startswith('static.prop.trafficcone') or x.id.startswith('static.prop.construction') or x.id.startswith('static.prop.barrel') or x.id.startswith('static.prop.bin')]

        bp_props = random.choice(bp_props)
        # PROCURA CALÇADAS PERTO DOS SPAWN POINTS
        transform = self.veh_spawn_points[props_num+self.ego_vehicle_num+self.npc_vehicle_num]
        map = self.world.get_map()
        waypoint = map.get_waypoint(transform.location, project_to_road=True,
                                      lane_type=(carla.LaneType.Sidewalk))

        bp_props.set_attribute('role_name', 'PROP ' + str(props_num + 1))  # EGO 1, EGO 2
        #bp_props.set_attribute('semantic_tags', 3)
        self.static_props.append(self.world.spawn_actor(bp_props, waypoint.transform))
        self.world.wait_for_tick()
        time.sleep(0.1)
        #self.static_props[props_num].set("semantic_tags", 3)  # PENSAR EM COMO FAZER
        self.props_list.append(self.static_props[props_num].id)

        # Muda a visão para o local onde aconteceu o spawn - APENAS DEBUG
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)

    def spawn_all(self):
        # ============== SPAWN DE PEDESTRES, VEÍCULOS E OBJETOS ==============
        self.spawn_pedestrians()  # CARREGA PEDESTRES

        for veh_num in range(self.ego_vehicle_num):  # FAZ SPAWN DOS CARROS EGO
            self.spawn_vehicle(veh_num, True)  # CARREGA OS VEÍCULOS NA SIMULAÇÃO
        # registra os eventos em formato de log
        print("Criado(s) %s veículo(s) EGO" % self.ego_vehicle_num)

        for veh_num in range(self.npc_vehicle_num):  # FAZ SPAWN DOS CARROS NPCS
            self.spawn_vehicle(veh_num, False)
        # registra os eventos em formato de log
        print("Criado(s) %s veículo(s) NPC" % self.npc_vehicle_num)

        for props_num in range(self.static_props_num):  # FAZ SPAWN DE OBJETOS ESTÁTICOS
            self.spawn_props(props_num)
        # registra os eventos em formato de log
        print("Criado(s) %s obstáculo(s)" % self.static_props_num)

# CLASSE USADA PARA IMPLEMENTAR O SENSOR DE "DIREÇÃO"
class Speed_SAS_Sensor(object):
  def __init__(self, interval, function, *args, **kwargs):
    self._timer = None
    self.interval = interval
    self.function = function
    self.args = args
    self.kwargs = kwargs
    self.is_running = False
    self.next_call = time.time()
    self.start()

  def _run(self):
    self.is_running = False
    self.start()
    self.function(*self.args, **self.kwargs)

  def start(self):
    if not self.is_running:
      self.next_call += self.interval
      self._timer = Timer(self.next_call - time.time(), self._run)
      self._timer.start()
      self.is_running = True

  def stop(self):
    self._timer.cancel()
    self.is_running = False


# CLASSE USADA PARA PAUSAR E RESUMIR A SIMULAÇÃO DURANTE O SETUP
class SimPause(object):
    def __init__(self):
        self._SimulationSetup = None
        self.settings = None
        self.traffic_manager = None

    def start(self, sim):
        """Assigns other initialized modules that input module needs."""
        self._SimulationSetup = sim
        self.settings = sim.world.get_settings()
        self.traffic_manager = sim.traffic_manager

    def pause(self, sim):
        # CONFIGURA A SIMULAÇÃO/TRAFFIC_MANAGER EM MODO SÍNCRONO PARA PAUSAR A EXECUÇÃO DOS TESTES
        #self.settings.synchronous_mode = True
        #self.settings.fixed_delta_seconds = 1.0 / 20
        #self.traffic_manager.set_synchronous_mode(True)
        #self._SimulationSetup.world.apply_settings(self.settings)
        #self._SimulationSetup.world.tick()
        for veh in sim.ego_vehicle:
            veh.agent.set_target_speed(0)
            #veh.apply_control(control)
        for veh in sim.npc_vehicle:
            veh.agent.set_target_speed(0)
            #veh.apply_control(control)

    def resume(self, sim, sim_params):
        # CONFIGURA A SIMULAÇÃO/TRAFFIC_MANAGER EM MODO ASSÍNCRONO PARA RESUMIR A EXECUÇÃO DOS TESTES
        #self.settings.synchronous_mode = False
        #self.settings.fixed_delta_seconds = 1.0 / 20
        #self.traffic_manager.set_synchronous_mode(False)
        #self._SimulationSetup.world.apply_settings(self.settings)
        #self._SimulationSetup.world.tick()
        speed = sim_params["VEHICLE_SPEED"]
        behavior = sim_params["VEHICLE_BEHAVIOR"]
        agent = sim_params["VEHICLE_AGENT"]

        if agent == "BASIC" and speed == "Limit" and behavior == "randomized":
            for veh in sim.ego_vehicle:
                max_speed = random.randrange(40, 50, 1)
                print(max_speed)
                veh.agent.set_target_speed(max_speed)
                #veh.apply_control(control)
            for veh in sim.npc_vehicle:
                max_speed = random.randrange(40, 50, 1)
                print(max_speed)
                veh.agent.set_target_speed(max_speed)
                #veh.apply_control(control)
        elif agent == "BASIC" and speed == "Limit" and behavior != "randomized":
            for veh in sim.ego_vehicle:
                veh.agent.set_target_speed(speed)
            for veh in sim.npc_vehicle:
                veh.agent.set_target_speed(speed)
        else:
            pass

# CONFIGURA A VISÃO SUPERIOR DO MAPA
class TopView(object):
    def __init__(self,sim_params):
        # self.classes  # importa no_rendering_mode.py
        self.input_control = None
        self.hud = None
        self.world = None
        self.display = None
        self.clock = None
        self.first_start = True

        # Unpack variáveis
        self.screen_width = sim_params["SCREEN_WIDTH"]
        self.screen_height = sim_params["SCREEN_HEIGHT"]
        self.top_view_show_hud = sim_params["TOP_VIEW_SHOW_HUD"]
        self.top_view_show_id = sim_params["TOP_VIEW_SHOW_ID"]
        self.config_fps = sim_params["CONFIG_FPS"]
        self.last_positions_training = sim_params["LAST_POSITIONS_TRAINING"]

    # ======== INICIALIZA OS MÓDULOS NECESSÁRIOS PARA CONFIGURAR A VISÃO DE CIMA =========
    def start(self, mapa):

        # Init Pygame
        if self.first_start:
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)

            # Place a title to game window
            pygame.display.set_caption("Visão Superior")

            # Carrega Pygame apenas uma vez
            self.first_start = False

        # Show loading screen
        font = pygame.font.Font(pygame.font.get_default_font(), 20)

        text_surface = font.render('Renderizando Mapa...', True, top_view.COLOR_WHITE)
        self.display.blit(text_surface, text_surface.get_rect(center=(self.screen_width / 2, self.screen_height / 2)))
        pygame.display.flip()

        # Init classes
        self.input_control = top_view.InputControl(top_view.TITLE_INPUT)
        self.hud = top_view.HUD(top_view.TITLE_HUD, self.screen_width, self.screen_height, self.top_view_show_hud, self.top_view_show_id)
        self.world = top_view.World(top_view.TITLE_WORLD)

        # For each module, assign other modules that are going to be used inside that module
        self.input_control.start(self.hud, self.world)
        self.hud.start()
        self.world.start(self.hud, self.input_control, mapa)

        self.clock = pygame.time.Clock()
        # self.tick()

    # ========= ATUALIZA O DISPLAY TODA VEZ QUE RODAR (COLOCAR DENTRO DE UM LOOP) =============
    def tick(self, hud_text, _all_veh = None):
        self.clock.tick_busy_loop(self.config_fps)

        # Tick all modules
        self.world.tick(self.clock, hud_text)  # hud_txt insere texto no hud - Formato ["Título_Texto1;Texto2"]
        self.hud.tick(self.clock)
        self.input_control.tick(self.clock)

        # Render all modules
        if self.input_control.quit == False:
            self.display.fill(top_view.COLOR_ALUMINIUM_4)
            self.world.render(self.display, _all_veh)
            self.hud.render(self.display)
            self.input_control.render(self.display)
            self.world.ground_truth(self.display, _all_veh, self.last_positions_training)

            pygame.display.flip()