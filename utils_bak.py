import types
import cv2
import numpy as np
import scipy.signal
import tensorflow as tf


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):

        # inicializa o recorder
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"XVID"), int(fps),
            frame_size)
            #(frame_size[1], frame_size[0]))


    def add_frame(self, frame):
        #cv2.imshow("teste",frame)
        #print("frame: ", frame)
        self.video_writer.write(frame)
        #pygame.surfarray.array2d(img)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def create_counter_variable(name):
    counter = types.SimpleNamespace()
    counter.var = tf.Variable(0, name=name, trainable=False)
    counter.inc_op = tf.assign(counter.var, counter.var + 1)
    return counter

def create_mean_metrics_from_dict(metrics):
    # Set up summaries for each metric
    update_metrics_ops = []
    summaries = []
    for name, (value, update_op) in metrics.items():
        summaries.append(tf.summary.scalar(name, value))
        update_metrics_ops.append(update_op)
    return tf.summary.merge(summaries), tf.group(update_metrics_ops)

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrap_values])
    terminals = np.array(terminals)
    #print(rewards)
    #print(values)

    #print("rewards: ",rewards)
    #print("terminals: ",terminals)
    #print("gamma: ",gamma)
    #print("values: ", values)

    deltas = rewards + (1.0 - terminals) * gamma * values[1:] - values[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]

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
#dt = 0.1
#var_pos = 0.1
#var_acc = 1.0
#kf = KalmanFilter(dt, var_pos, var_acc)

# Define a posição inicial e a aceleração inicial do veículo
#posX = random.uniform(-5, 5)
#posY = random.uniform(-5, 5)
#acelX = random.uniform(-2, 2)
#acelY = random.uniform(-2, 2)

# Faz as medições e predições da posição do veículo
#for i in range(100):
    # Simula a medição da posição
#    posX += acelX*dt**2/2 + random.gauss(0, var_pos)
#    posY += acelY*dt**2/2 + random.gauss(0, var_pos)
#    z = np.array([[posX], [posY]])

    # Simula a aceleração do veículo
#    acelX += random.gauss(0, var_acc)
#    acelY += random.gauss(0, var_acc)

    # Executa o filtro de Kalman para prever a posição do veículo
#    predX, predY = kf.run(z)

    # Imprime a posição medida, a posição prevista e a aceleração atual
#    print(f"Medição: ({posX:.2f}, {posY:.2f}) - Predição: ({predX[0]:.2f}, {predY[0]:.2f}) - Aceleração: ({acelX:.2f}, {acelY:.2f})")