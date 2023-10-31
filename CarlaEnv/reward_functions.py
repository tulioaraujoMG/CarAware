import numpy as np
from wrappers import angle_diff, vector

#low_speed_timer = 0
#max_distance    = 3.0  # Max distance from center before terminating
#target_speed    = 20.0 # kmh


def calculate_reward(env, reward_fn, last_reward = None, last_distance_lst = None, veh = None, veh_num = None):

    if reward_fn == "rw_distance":
        reward, distance = rw_distance(env)

    if reward_fn == "rw_distance_normalized":
        reward, distance = rw_distance_normalized(env, veh, veh_num)

    if reward_fn == "rw_distance_with_high_penalization":
        total_reward, distance = rw_distance_with_high_penalization(env)
        if total_reward >= last_reward:
            reward = total_reward
        else:
            reward = -total_reward

    if reward_fn == "rw_distance_with_low_penalization":  # distance_mean = sum(distance_lst)/env.ego_num
        close_reward_lst, distance = rw_distance_with_low_penalization(env)  # close_reward_lst,
        reward = 0
        for current_distance, last_distance in zip(distance, last_distance_lst):
            if current_distance <= last_distance:
                reward += 10
            else:
                reward -= 5
        reward += sum(close_reward_lst)/env.ego_num

    if reward_fn == "rw_exponential_distance_normalized":
        reward, distance = rw_exponential_distance_normalized(env, veh, veh_num)

        #reward = reward/env.ego_num (verificar se faz sentido trabalhar com a média pra vários veículos

    return reward, distance


def rw_distance_with_low_penalization(env):

    # CALCULA DISTÂNCIA ENTRE OS PONTOS DEFINIDOS PELA RN E O GT
    distance_lst = []
    close_reward_lst = []
    try:
        for vehicle_pred, vehicle_gt in zip(env._simulation.ego_vehicle, env._top_view.world.gt_input_ego):

            distance = np.sqrt((vehicle_pred.prediction[0] - vehicle_gt.x) ** 2 + (
                    vehicle_pred.prediction[1] - vehicle_gt.y) ** 2)

            close_reward = 308 / (distance + 1)
            if 15 < distance < 30:
                close_reward = close_reward*1.5
            elif distance < 15:
                close_reward = close_reward*2

            close_reward_lst.append(close_reward)
            distance_lst.append(distance)
    except:
        close_reward_lst.append(0)
        distance_lst.append(0)

    return close_reward_lst, distance_lst


def rw_distance_with_high_penalization(env):

    # CALCULA DISTÂNCIA ENTRE OS PONTOS DEFINIDOS PELA RN E O GT
    reward_lst = []
    distance_lst = []

    try:
        for vehicle_pred, vehicle_gt in zip(env._simulation.ego_vehicle, env._top_view.world.gt_input_ego):

            distance = np.sqrt((vehicle_pred.prediction[0] - vehicle_gt.x) ** 2 + (
                    vehicle_pred.prediction[1] - vehicle_gt.y) ** 2)

            reward = 308 / (distance + 1)  # 308 é a distância máxima de canto a canto do mapa -> sqrt((210+15)^2+(100-310)^2)
            reward_lst.append(reward)
            distance_lst.append(distance)
    except:
        reward_lst.append(0)
        distance_lst.append(0)

    total_reward = sum(reward_lst)/env.ego_num
    return total_reward, distance_lst


def rw_distance(env):

    # CALCULA DISTÂNCIA ENTRE OS PONTOS DEFINIDOS PELA RN E O GT
    reward_lst = []
    distance_lst = []

    errors = 0
    try:

        for vehicle_pred, vehicle_gt in zip(env._simulation.ego_vehicle, env._top_view.world.gt_input_ego):

            distance = np.sqrt((vehicle_pred.prediction[0] - vehicle_gt.x) ** 2 + (
                    vehicle_pred.prediction[1] - vehicle_gt.y) ** 2)

            reward = 308 / (distance + 1)  # 308 é a distância máxima de canto a canto do mapa -> sqrt((210+15)^2+(100-310)^2)
            reward_lst.append(reward)
            distance_lst.append(distance)

    except:
        errors += 1
    #    reward_lst.append(0)
    #    distance_lst.append(0)

    total_reward = sum(reward_lst)/(env.ego_num-errors)
    return total_reward, distance_lst

def rw_exponential_distance_normalized(env, veh, veh_num):

    # CALCULA DISTÂNCIA ENTRE OS PONTOS DEFINIDOS PELA RN E O GT
    reward_lst = []
    distance_lst = []

    try:
        veh_gt = env._top_view.world.gt_input_ego
        distance = np.sqrt((veh.prediction[0] - veh_gt[veh_num].x) ** 2 + (
                veh.prediction[1] - veh_gt[veh_num].y) ** 2)

        reward = 1 / np.exp(distance)

            #reward_lst.append(reward)
            #distance_lst.append(distance)
    except:
        #reward_lst.append(0)
        #distance_lst.append(0)
        reward = 0
        distance = 0

    #total_reward = sum(reward_lst)/env.ego_num
    return reward, distance

def rw_distance_normalized(env, veh, veh_num):

    # CALCULA DISTÂNCIA ENTRE OS PONTOS DEFINIDOS PELA RN E O GT
    #reward_lst = []
    #distance_lst = []

    try:
        veh_gt = env._top_view.world.gt_input_ego
        distance = np.sqrt((veh.prediction[0] - veh_gt[veh_num].x) ** 2 + (
                veh.prediction[1] - veh_gt[veh_num].y) ** 2)


        reward = 1 / distance
        #reward_lst.append(reward)
        #distance_lst.append(distance)
    except:
        reward = 1
        distance = 0
        #reward_lst.append(0)
        #distance_lst.append(0)

    #total_reward = sum(reward_lst)/env.ego_num

    return reward, distance