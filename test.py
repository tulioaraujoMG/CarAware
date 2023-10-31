#sim_total_time = 11

#print(sim_total_time // 3600)
#print(sim_total_time % 3600 // 60)
#print(sim_total_time % 60)

#import gym
#import numpy as np
#import tensorflow as tf
#import math
#import carla
#import random
#import time
#from time import sleep

'''
#action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

action_space = {"Obj_Coord": gym.spaces.Box(np.array([-20, 220]), np.array([80, 320]), dtype=np.int32), "Obj_Type":
    gym.spaces.Discrete(3), "Obj_Num":gym.spaces.Discrete(2)}

print(action_space.values())

action_space = gym.spaces.Dict({"Obj_Coord": gym.spaces.Box(np.array([-20, 220]), np.array([80, 320]), dtype=np.int32),
                     "Obj_Type": gym.spaces.Discrete(3), "Obj_Num": gym.spaces.Discrete(2)})

print(action_space.items())
#action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
#num_actions = len(action_space)
#num_actions = action_space.shape[0]
#print(num_actions)

#observation_space = {"GNSS": gym.spaces.Box(np.array([-20, 220]), np.array([80, 320]), dtype=np.int32),
#                        "Obj_Type": gym.spaces.Discrete(3)}

obs_res=(1280, 720)
print(*obs_res)
observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)

input_shape = observation_space.shape
#input_shape = observation_space["GNSS"].shape[0] +2
#input_shape = len(observation_space)+2
print(input_shape)

input_states  = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
'''

# VERIFICA POSIÇÕES NO TOP-VIEW
'''
x_ini = -20
y_ini = 90

x_fim = 210
y_fim = 320

location_ini = top_view.world.map_image.world_to_pixel(carla.Location(x=x_ini, y=y_ini))
print("location ini: ", location_ini)
location_fim = top_view.world.map_image.world_to_pixel(carla.Location(x=x_fim, y=y_fim))
print("location fim: ", location_fim)
'''

'''
MAP = "Town02"
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
client.load_world(MAP, True)
world.wait_for_tick()
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.global_percentage_speed_difference(30.0)
#set_autopilot = carla.command.SetAutopilot

veh_spawn_points = world.get_map().get_spawn_points()

bp_vehicle = world.get_blueprint_library().filter("vehicle.*")
bp_vehicle = [x for x in bp_vehicle if int(x.get_attribute('number_of_wheels')) == 4]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('isetta')]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('carlacola')]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('cybertruck')]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('firetruck')]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('t2')]
bp_vehicle = [x for x in bp_vehicle if not x.id.endswith('ambulance')]
bp_vehicle = sorted(bp_vehicle, key=lambda bp: bp.id)
bp_vehicle = random.choice(bp_vehicle)

npc_vehicle = []
vehicles_list = []

transform = veh_spawn_points[0]  # pula os spawn points usados pelos EGO
npc_vehicle.append(world.spawn_actor(bp_vehicle, transform))
world.wait_for_tick()
time.sleep(0.1)
npc_vehicle[0].set_autopilot(True, traffic_manager.get_port())

transform = veh_spawn_points[1]  # pula os spawn points usados pelos EGO
npc_vehicle.append(world.spawn_actor(bp_vehicle, transform))
world.wait_for_tick()
time.sleep(0.1)
npc_vehicle[1].set_autopilot(True, traffic_manager.get_port())

transform = veh_spawn_points[2]  # pula os spawn points usados pelos EGO
npc_vehicle.append(world.spawn_actor(bp_vehicle, transform))
world.wait_for_tick()
time.sleep(0.1)
npc_vehicle[2].set_autopilot(True, traffic_manager.get_port())

while True:
    world.wait_for_tick()
    time.sleep(1)

'''
#m = np.mean([1,2,3])
#print()

'''
import threading
import time

class RepeatedTimer(object):
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
      self._timer = threading.Timer(self.next_call - time.time(), self._run)
      self._timer.start()
      self.is_running = True

  def stop(self):
    self._timer.cancel()
    self.is_running = False

def hello(name):
    print("Hello %s!" % name)

print("starting")
rt = RepeatedTimer(1, hello, "World")  # it auto-starts, no need of rt.start()
try:
    sleep(20)  # your long-running job goes here...
finally:
    rt.stop()  # better in a try/finally block to make sure the program ends!

'''
#ego_num = 2
#for veh in range(ego_num):
  #print(veh)
'''
import numpy as np
import random

class KalmanFilter:
    def __init__(self, dt, var_pos, var_acc):
        self.dt = dt
        self.var_pos = var_pos
        self.var_acc = var_acc
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                           [0, dt**4/4, 0, dt**3/2],
                           [dt**3/2, 0, dt**2, 0],
                           [0, dt**3/2, 0, dt**2]]) * var_acc
        self.R = np.array([[var_pos, 0],
                           [0, var_pos]])
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)

    def run(self, z):
        self.predict()
        self.update(z)
        return self.x[0], self.x[1]


# Cria um objeto KalmanFilter
dt = 0.1
var_pos = 0.1
var_acc = 1.0
kf = KalmanFilter(dt, var_pos, var_acc)

# Define a posição inicial e a aceleração inicial do veículo
posX = random.uniform(-5, 5)
posY = random.uniform(-5, 5)
acelX = random.uniform(-2, 2)
acelY = random.uniform(-2, 2)

# Faz as medições e predições da posição do veículo
for i in range(100):
    # Simula a medição da posição
    posX += acelX*dt**2/2 + random.gauss(0, var_pos)
    posY += acelY*dt**2/2 + random.gauss(0, var_pos)
    z = np.array([[posX], [posY]])

    # Simula a aceleração do veículo
    acelX += random.gauss(0, var_acc)
    acelY += random.gauss(0, var_acc)

    # Executa o filtro de Kalman para prever a posição do veículo
    predX, predY = kf.run(z)

    # Imprime a posição medida, a posição prevista e a aceleração atual
    print(f"Medição: ({posX:.2f}, {posY:.2f}) - Predição: ({predX[0]:.2f}, {predY[0]:.2f}) - Aceleração: ({acelX:.2f}, {acelY:.2f})")

'''
import os
csv_file = open(os.path.join("C:\\Users\\tulioaraujo\\PycharmProjects\\carla\\models\\PPO_MODEL_step5_moving_1agent_reset10_gradualrandom_distnorm_noblackout_highstd_h32768_batch2048_lr5e8_epoch4_v1\\", "training_log.csv"), "r")
try:
    data = [line.strip().split(';')[5] for line in csv_file.readlines()]
except:
    print("não existe")
csv_file.close()