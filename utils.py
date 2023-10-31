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

class KalmanFilter_old:
    def __init__(self, dt, var_pos, var_acc):
        self.dt = dt
        self.var_pos = var_pos
        self.var_acc = var_acc
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        #self.H = np.array([[1, 0, 0, 0],
                           # [0, 1, 0, 0]])
        self.H = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1]])
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


# Utitlity file with functions for handling rotations.
#
# Authors: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
# Credit: taken from State Estimation and Localization for Self-Driving Cars by Coursera
#   Please consider enrolling the course if you find this tutorial helpful and
#   would like to learn more about Kalman filter and state estimation for
#   self-driving cars in general.

def angle_normalize(a):
    """Normalize angles to lie in range -pi < a[i] <= pi."""
    a = np.remainder(a, 2*np.pi)
    a[a <= -np.pi] += 2*np.pi
    a[a  >  np.pi] -= 2*np.pi
    return a

def skew_symmetric(v):
    """Skew symmetric form of a 3x1 vector."""
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]], dtype=np.float64)

def rpy_jacobian_axis_angle(a):
    """Jacobian of RPY Euler angles with respect to axis-angle vector."""
    if not (type(a) == np.ndarray and len(a) == 3):
        raise ValueError("'a' must be a np.ndarray with length 3.")
    # From three-parameter representation, compute u and theta.
    na  = np.sqrt(a @ a)
    na3 = na**3
    t = np.sqrt(a @ a)
    u = a/t

    # First-order approximation of Jacobian wrt u, t.
    Jr = np.array([[t/(t**2*u[0]**2 + 1), 0, 0, u[0]/(t**2*u[0]**2 + 1)],
                   [0, t/np.sqrt(1 - t**2*u[1]**2), 0, u[1]/np.sqrt(1 - t**2*u[1]**2)],
                   [0, 0, t/(t**2*u[2]**2 + 1), u[2]/(t**2*u[2]**2 + 1)]])

    # Jacobian of u, t wrt a.
    Ja = np.array([[(a[1]**2 + a[2]**2)/na3,        -(a[0]*a[1])/na3,        -(a[0]*a[2])/na3],
                   [       -(a[0]*a[1])/na3, (a[0]**2 + a[2]**2)/na3,        -(a[1]*a[2])/na3],
                   [       -(a[0]*a[2])/na3,        -(a[1]*a[2])/na3, (a[0]**2 + a[1]**2)/na3],
                   [                a[0]/na,                 a[1]/na,                 a[2]/na]])

    return Jr @ Ja

def omega(w, delta_t):
    """
    Purturbation locally at the SO3 manifold
    (right quaternion product matrix)
    """
    theta = w * delta_t
    q = Quaternion(axis_angle=angle_normalize(theta)).to_numpy()
    qw = q[0]
    qv = q[1:]

    om1 = np.zeros((4, 4))
    om1[0, 1:] = -qv
    om1[1:, 0] = qv
    om1[1:, 1:] = -skew_symmetric(qv)
    om = np.identity(4) * qw + om1

    return om

class Quaternion():
    def __init__(self, w=1., x=0., y=0., z=0., axis_angle=None, euler=None):
        """
        Allow initialization with explicit quaterion wxyz, axis-angle, or Euler XYZ (RPY) angles.

        :param w: w (real) of quaternion.
        :param x: x (i) of quaternion.
        :param y: y (j) of quaternion.
        :param z: z (k) of quaternion.
        :param axis_angle: Set of three values from axis-angle representation, as list or [3,] or [3,1] np.ndarray.
                           See C2M5L2 for details.
        :param euler: Set of three XYZ Euler angles.
        """
        if axis_angle is None and euler is None:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        elif euler is not None and axis_angle is not None:
            raise AttributeError("Only one of axis_angle or euler can be specified.")
        elif axis_angle is not None:
            if not (type(axis_angle) == list or type(axis_angle) == np.ndarray) or len(axis_angle) != 3:
                raise ValueError("axis_angle must be list or np.ndarray with length 3.")
            axis_angle = np.array(axis_angle)
            norm = np.linalg.norm(axis_angle)
            self.w = np.cos(norm / 2)
            if norm < 1e-50:  # to avoid instabilities and nans
                self.x = 0
                self.y = 0
                self.z = 0
            else:
                imag = axis_angle / norm * np.sin(norm / 2)
                self.x = imag[0].item()
                self.y = imag[1].item()
                self.z = imag[2].item()
        else:
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)

            # Fixed frame
            self.w = cr * cp * cy + sr * sp * sy
            self.x = sr * cp * cy - cr * sp * sy
            self.y = cr * sp * cy + sr * cp * sy
            self.z = cr * cp * sy - sr * sp * cy

            # Rotating frame
            # self.w = cr * cp * cy - sr * sp * sy
            # self.x = cr * sp * sy + sr * cp * cy
            # self.y = cr * sp * cy - sr * cp * sy
            # self.z = cr * cp * sy + sr * sp * cy

    def __repr__(self):
        return "Quaternion (wxyz): [%2.5f, %2.5f, %2.5f, %2.5f]" % (self.w, self.x, self.y, self.z)

    def to_axis_angle(self):
        t = 2*np.arccos(self.w)
        return np.array(t*np.array([self.x, self.y, self.z])/np.sin(t/2))

    def to_mat(self):
        v = np.array([self.x, self.y, self.z]).reshape(3,1)
        return (self.w ** 2 - np.dot(v.T, v)) * np.eye(3) + \
               2 * np.dot(v, v.T) + 2 * self.w * skew_symmetric(v)

    def to_euler(self):
        """Return as xyz (roll pitch yaw) Euler angles."""
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))
        return np.array([roll, pitch, yaw])

    def to_numpy(self):
        """Return numpy wxyz representation."""
        return np.array([self.w, self.x, self.y, self.z])

    def normalize(self):
        """Return a (unit) normalized version of this quaternion."""
        norm = np.linalg.norm([self.w, self.x, self.y, self.z])
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def quat_mult_right(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the right, that is, q*self.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = -skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

    def quat_mult_left(self, q, out='np'):
        """
        Quaternion multiplication operation - in this case, perform multiplication
        on the left, that is, self*q.

        :param q: Either a Quaternion or 4x1 ndarray.
        :param out: Output type, either np or Quaternion.
        :return: Returns quaternion of desired type.
        """
        v = np.array([self.x, self.y, self.z]).reshape(3, 1)
        sum_term = np.zeros([4,4])
        sum_term[0,1:] = -v[:,0]
        sum_term[1:, 0] = v[:,0]
        sum_term[1:, 1:] = skew_symmetric(v)
        sigma = self.w * np.eye(4) + sum_term

        if type(q).__name__ == "Quaternion":
            quat_np = np.dot(sigma, q.to_numpy())
        else:
            quat_np = np.dot(sigma, q)

        if out == 'np':
            return quat_np
        elif out == 'Quaternion':
            quat_obj = Quaternion(quat_np[0], quat_np[1], quat_np[2], quat_np[3])
            return quat_obj

# Author: Shing-Yan Loo (yan99033@gmail.com)
# Run extended Kalman filter to calculate the real-time vehicle location
#
# Credit: State Estimation and Localization for Self-Driving Cars by Coursera
#   Please consider enrolling the course if you find this tutorial helpful and
#   would like to learn more about Kalman filter and state estimation for
#   self-driving cars in general.

from matplotlib.pyplot import axis

class ExtendedKalmanFilter:
    def __init__(self):
        # State (position, velocity and orientation)
        self.p = np.zeros([3, 1])
        self.v = np.zeros([3, 1])
        self.q = np.zeros([4, 1])  # quaternion

        # State covariance
        self.p_cov = np.zeros([9, 9])

        # Last updated timestamp (to compute the position
        # recovered by IMU velocity and acceleration, i.e.,
        # dead-reckoning)
        self.last_ts = 0

        # Gravity
        self.g = np.array([0, 0, -9.81]).reshape(3, 1)

        # Sensor noise variances
        self.var_imu_acc = 0.01  # valor original: 0.01 - melhor: 10
        self.var_imu_gyro = 0.01  # valor original: 0.01 - melhor: 10

        # Motion model noise
        self.var_gnss = np.eye(3) * 0.1  # valor original: 100

        # Motion model noise Jacobian
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        # Measurement model Jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)

        # Initialized
        self.n_gnss_taken = 0
        self.gnss_init_xyz = None
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def initialize_with_gnss(self, gnss, samples_to_use=5):  #samples_to_use = 10 (default)
        """Initialize the vehicle state using gnss sensor

        Note that this is going to be a very crude initialization by taking
        an average of 10 readings to get the absolute position of the car. A
        better initialization technique could be employed to better estimate
        the initial vehicle state

        Alternatively, you can also initialize the vehicle state using ground
        truth vehicle position and orientation, but this would take away the
        realism of the experiment/project

        :param gnss: converted absolute xyz position
        :type gnss: list
        """
        if self.gnss_init_xyz is None:
            self.gnss_init_xyz = np.array([gnss[0], gnss[1], gnss[2]])
        else:
            self.gnss_init_xyz[0] += gnss[0]
            self.gnss_init_xyz[1] += gnss[1]
            self.gnss_init_xyz[2] += gnss[2]
        self.n_gnss_taken += 1

        if self.n_gnss_taken == samples_to_use:
            self.gnss_init_xyz /= samples_to_use
            self.p[:, 0] = self.gnss_init_xyz
            self.q[:, 0] = Quaternion().to_numpy()

            # Low uncertainty in position estimation and high in orientation and
            # velocity
            pos_var = 1000  # original: 1
            orien_var = 1  # original: 1000
            vel_var = 1  # original: 1000
            self.p_cov[:3, :3] = np.eye(3) * pos_var
            self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
            self.p_cov[6:, 6:] = np.eye(3) * orien_var
            self.initialized = True


    def get_location(self):
        """Return the estimated vehicle location

        :return: x, y, z position
        :rtype: list
        """
        return self.p.reshape(-1).tolist()

    def predict_state_with_imu(self, imu):
        """Use the IMU reading to update the car location (dead-reckoning)

        (This is equivalent to doing EKF prediction)

        Note that if the state is just initialized, there might be an error
        in the orientation that leads to incorrect state prediction. The error
        could be aggravated due to the fact that IMU is 'strapped down', and hence
        generating relative angular measurement (instead of absolute using IMU
        stabilized by a gimbal). Learn more in the Coursera course!

        The uncertainty (or state covariance) is going to grow larger and larger if there
        is no correction step. Therefore, the GNSS update would have a larger weight
        when performing the correction, and hopefully the state would converge toward
        the true state with more correction steps.

        :param imu: imu acceleration, velocity and timestamp
        :type imu: IMU blueprint instance (Carla)
        """
        # IMU acceleration and velocity
        imu_f = np.array([imu.ue_accelerometer[0], imu.ue_accelerometer[1], imu.ue_accelerometer[2]]).reshape(3, 1)
        imu_w = np.array([imu.ue_gyroscope[0], imu.ue_gyroscope[1], imu.ue_gyroscope[2]]).reshape(3, 1)

        # IMU sampling time
        delta_t = imu.timestamp - self.last_ts
        self.last_ts = imu.timestamp

        # Update state with imu
        R = Quaternion(*self.q).to_mat()
        self.p = self.p + delta_t * self.v + 0.5 * delta_t * delta_t * (R @ imu_f + self.g)
        self.v = self.v + delta_t * (R @ imu_f + self.g)
        self.q = omega(imu_w, delta_t) @ self.q

        # Update covariance
        F = self._calculate_motion_model_jacobian(R, imu_f, delta_t)
        Q = self._calculate_imu_noise(delta_t)
        self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T

    def _calculate_motion_model_jacobian(self, R, imu_f, delta_t):
        """derivative of the motion model function with respect to the state

        :param R: rotation matrix of the state orientation
        :type R: NumPy array
        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:] = -skew_symmetric(R @ imu_f) * delta_t

        return F

    def _calculate_imu_noise(self, delta_t):
        """Calculate the IMU noise according to the pre-defined sensor noise profile

        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        :param imu_w: IMU xyz angular velocity
        :type imu_w: NumPy array
        """
        Q = np.eye(6)
        Q[:3, :3] *= delta_t * delta_t * self.var_imu_acc
        Q[3:, 3:] *= delta_t * delta_t * self.var_imu_gyro

        return Q

    def correct_state_with_gnss(self, gnss):
        """Given the estimated global location by gnss, correct
        the vehicle state

        :param gnss: global xyz position
        :type x: Gnss class (see car.py)
        """
        # Global position
        x = gnss[0]
        y = gnss[1]
        z = gnss[2]

        # Kalman gain
        K = self.p_cov @ self.h_jac.T @ (np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + self.var_gnss))

        # Compute the error state
        delta_x = K @ (np.array([x, y, z])[:, None] - self.p)

        # Correction
        self.p = self.p + delta_x[:3]
        self.v = self.v + delta_x[3:6]
        delta_q = Quaternion(axis_angle=angle_normalize(delta_x[6:]))
        self.q = delta_q.quat_mult_left(self.q)

        # Corrected covariance
        self.p_cov = (np.identity(9) - K @ self.h_jac) @ self.p_cov
