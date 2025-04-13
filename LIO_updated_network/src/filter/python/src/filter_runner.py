"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker.py
"""

import json

from numba import jit
import numpy as np

from filter.python.src.meas_source_network import MeasSourceNetwork
from filter.python.src.net_input_utils import NetInputBuffer, ImuCalib
from filter.python.src.scekf import ImuMSCKF
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.math_utils import mat_exp
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterRunner:
    """
    FilterRunner is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning,
        imu_calib_dic=None,
        force_cpu=False,
    ):
        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["sampling_freq"]
            config_from_network["window_time"] = data_json["window_time"]

        # frequencies and sizes conversion
        self.imu_freq_net = config_from_network.imu_freq_net  # imu frequency as input to the network
        window_size = int(
            (config_from_network.window_time * config_from_network.imu_freq_net) )
        self.net_input_size = window_size

        # EXAMPLE :
        # if using 200 samples with step size 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if using 400 samples with 200 past data and clone_every_n_netimu_sample 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError("window_time cannot be represented by integer number of updates.")
        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(
            config_from_network.imu_freq_net / update_freq
        )  # network inference/filter update interval
        assert (
            config_from_network.imu_freq_net % update_freq == 0
        )  # imu frequency must be a multiple of update frequency
        self.update_distance_num_clone = int(
            config_from_network.window_time * update_freq
        )

        # time
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(1.0 / self.update_freq * 1e6)

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} (samples)"
        )
        logging.info("IMU / Thrust input to the network frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            "Filter update stride state number: %i" % self.update_distance_num_clone
        )
        logging.info(
            f"Interpolating IMU / Thrust measurements every {self.dt_interp_us} [us] for the network input"
        )

        # IMU initial calibration
        self.icalib = ImuCalib()
        self.icalib.from_dic(imu_calib_dic)
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning)

        self.meas_source = MeasSourceNetwork(model_path, force_cpu)

        self.inputs_buffer = NetInputBuffer()

        # This callback is called at first update to initialize the filter
        self.callback_first_update = None

        # keep track of past timestamp and measurement
        self.last_t_us, self.last_acc, self.last_gyr, self.last_thrust = -1, None, None, None
        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False

    # Note, imu meas for the net are calibrated with offline calibration.
    @jit(forceobj=True, parallel=False, cache=False)
    def _get_inputs_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_ts_begin = t_begin_us
        net_ts_end = t_end_us - self.dt_interp_us

        # net_fn are either accel in imu frame or thrusts in imu frame
        net_fn, net_gyr, net_t_us = self.inputs_buffer.get_data_from_to(
            net_ts_begin, net_ts_end
        )

        assert net_gyr.shape[0] == self.net_input_size
        assert net_fn.shape[0] == self.net_input_size
        # get data from filter
        R_oldest_state_wfb, _, _ = self.filter.get_past_state(t_oldest_state_us)  # 3 x 3

        # dynamic rotation integration using filter states
        # Rs_net will contains delta rotation since t_begin_us
        Rs_bofbi = np.zeros((net_t_us.shape[0], 3, 3))  # N x 3 x 3
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_t_us.shape[0]):
            dt_us = net_t_us[j] - net_t_us[j - 1]
            dt = from_usec_to_sec(dt_us)
            dR = mat_exp(net_gyr[j, :].reshape((3, 1)) * dt)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        # find delta rotation index at time ts_oldest_state
        oldest_state_idx_in_net = np.where(net_t_us == t_oldest_state_us)[0][0]

        # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
        # so that Rs_net[idx] = R_oldest_state_wfb
        R_bofboldstate = (
            R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_fn_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_fn)  # N x 3
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N x 3
        net_t_s = from_usec_to_sec(net_t_us)

        return net_gyr_w, net_fn_w, net_t_s

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw, thrust=None):
        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw, thrust)
        else:
            logging.info(f"Initializing filter at time {t_us} [us]")
            if self.icalib:
                logging.info(f"Using bias from initial calibration")
                init_ba = self.icalib.accelBias
                init_bg = self.icalib.gyroBias
                # calibrate raw imu data
                acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                    acc_raw, gyr_raw) 
            else:
                logging.info(f"Using zero bias")
                init_ba = np.zeros((3,1))
                init_bg = np.zeros((3,1))
                acc_biascpst, gyr_biascpst = acc_raw, gyr_raw

            self.filter.initialize(acc_biascpst, t_us, init_ba, init_bg)
            self.next_interp_t_us = t_us
            self.next_aug_t_us = t_us
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, t_us)
            self.next_aug_t_us = t_us + self.dt_update_us
            self.last_t_us, self.last_acc, self.last_gyr = (
                t_us,
                acc_biascpst,
                gyr_biascpst,
            )
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw, thrust=None):
        """
        For new IMU measurement, after the filter has been initialized
        """
        # Eventually calibrate
        if self.icalib:
            # calibrate raw imu data with offline calibation
            # this is used for network feeding
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )

            # calibrate raw imu data with offline calibation scale
            # this is used for the filter. 
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_augmentation_and_update = t_us >= self.next_aug_t_us

        # if augmenting the state, check that we compute interpolated measurement also
        assert (
            do_augmentation_and_update and do_interpolation_of_imu
        ) or not do_augmentation_and_update, (
            "Augmentation and interpolation does not match!"
        )

        # augmentation propagation / propagation
        # propagate at IMU input rate, augmentation propagation depends on t_augmentation_s
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        # Inputs interpolation and data saving for network
        if do_interpolation_of_imu:
            self._add_interpolated_inputs_to_buffer(thrust, gyr_biascpst, t_us)
                
        self.filter.propagate(
            acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us
        )
        # filter update
        did_update = False
        if do_augmentation_and_update:
            did_update = self._process_update(t_us)
            # plan next update/augmentation of state
            self.next_aug_t_us += self.dt_update_us

        # set last value memory to the current one
        self.last_t_us, self.last_acc, self.last_gyr = t_us, acc_biascpst, gyr_biascpst
        self.last_thrust = thrust

        return did_update

    def _process_update(self, t_us):
        logging.debug(f"Upd. @ {t_us} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1 ]
        t_begin_us = t_oldest_state_us
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        # If we do not have enough IMU data yet, just wait for next time
        if t_begin_us < self.inputs_buffer.net_t_us[0]:
            return False
        # initialize with ground truth at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us

        # get measurement from network
        net_gyr_w, net_fn_w, net_t_s = self._get_inputs_samples_for_network(
            t_begin_us, t_oldest_state_us, t_end_us)
        meas, meas_cov = self.meas_source.get_displacement_measurement(
            net_t_s, net_gyr_w, net_fn_w)

        # filter update
        is_available, innovation, jac, noise_mat = \
            self.filter.learnt_model_update(meas, meas_cov, t_oldest_state_us, t_end_us)
        success = False
        if is_available:
            inno = innovation.reshape((3,1))
            success = self.filter.apply_update(inno, jac, noise_mat)

        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        logging.debug(f"marginalize {cut_idx}")
        self.filter.marginalize(cut_idx)
        self.inputs_buffer.throw_data_before(t_begin_us)
        return success

    def _add_interpolated_inputs_to_buffer(self, fn_in, gyr_biascpst, t_us):
        self.inputs_buffer.add_data_interpolated(
            self.last_t_us,
            t_us,
            self.last_gyr,
            gyr_biascpst,
            self.last_thrust,
            fn_in,
            self.next_interp_t_us,
        )

        self.next_interp_t_us += self.dt_interp_us


#   ---------------------------------------------------------------------------------------------------------------

# filter_tuning = dotdict(
#     g_norm=9.81,
#     imu_freq=100,
#     sigma_na=np.array([0.008, 0.025, 0.02]),  # X/Y/Z accel
#     sigma_ng=np.array([0.0008, 0.002, 0.0015]),  # X/Y/Z gyro
#     sigma_nba=np.array([0.00008, 0.0002, 0.00015]),
#     sigma_nbg=np.array([0.000008, 0.00002, 0.000015]),
#     init_pos_sigma=np.array([0.1, 1.8, 1.2]),
#     init_vel_sigma=np.array([0.2, 2.5, 1.8]),
#     init_attitude_sigma=np.array([0.01, 0.15, 0.12, 0.01])
# )


# class UKFInertialOdometry:

#     def __init__(self, filter_tuning):
#         # Initialize time tracking
#         self._last_t_us = 0 # Private backing field
#         self._last_update_time = 0
#         # Property definition
#         @property
#         def last_t_us(self):
#             return self._last_t_us  # Use correct variable name
        
#         @last_t_us.setter
#         def last_t_us(self, value):
#             self._last_t_us = value

#         # 1. Define state dimensions FIRST
#         self.state_dim = 16  # [pos(3) + vel(3) + quat(4) + acc_bias(3) + gyro_bias(3)]
#         self.meas_dim = 5    # Modified for pseudo-measurements
#         self.axis_weights = np.array([1.0, 0.5, 0.5])  # X,Y,Z weights

#         # 2. Initialize other attributes AFTER state_dim
#         self.filter_tuning = filter_tuning
#         self.dt = 1.0/filter_tuning.get('imu_freq', 100)
#         self.g_norm = filter_tuning.g_norm
        
#         # 3. Initialize UKF components using state_dim
#         self.sigma_points = MerweScaledSigmaPoints(
#             n=self.state_dim,  # Now properly defined
#             alpha=0.1,
#             beta=2.0,
#             kappa=3-self.state_dim
#         )
        
#         self.ukf = UnscentedKalmanFilter(
#             dim_x=self.state_dim,
#             dim_z=self.meas_dim,
#             dt=self.dt,
#             fx=self.state_transition,
#             hx=self.measurement_model,
#             points=self.sigma_points
#         )
        
#         # Rest of initialization
#         self.initialize_filter()
#         self.g = np.array([0, 0, -self.g_norm])
#         self.current_imu = {'acc': None, 'gyro': None}
#         self.initialized = False
#         # self.last_t_us = None


#     def state_transition(self, x, dt):
#         pos = x[0:3]
#         vel = x[3:6]
#         quat = x[6:10]
#         acc_bias = x[10:13]
#         gyro_bias = x[13:16]

#         # Improved bias compensation with moving average
#         acc = self.current_imu['acc'] - acc_bias
#         gyro = self.current_imu['gyro'] - gyro_bias

#         # Robust quaternion integration
#         delta_angle = gyro * dt
#         delta_q = Rotation.from_rotvec(delta_angle).as_quat()
#         new_rot = Rotation.from_quat(quat) * Rotation.from_quat(delta_q)
#         new_quat = new_rot.as_quat()
#         new_quat /= np.linalg.norm(new_quat) 
#         # Proper gravity handling with Earth frame alignment
#         R = new_rot.as_matrix()
#         gravity_world = R @ np.array([0, 0, -self.g_norm])
#         g_body = R.T @ np.array([0, 0, -self.g_norm])
#         acc_body = self.current_imu['acc'] - acc_bias - g_body
#         acc_world = R @ acc_body

        
#         # Velocity damping for Y/Z axes
#         vel_damping = np.array([1.0, 1.0, 0.95])
#         new_vel = vel * vel_damping + acc_world * dt
        
#         # Position update
#         new_pos = pos + new_vel * dt + 0.5 * acc_world * dt**2

#         return np.concatenate([
#             new_pos.ravel(),
#             new_vel.ravel(),
#             new_quat.ravel(),
#             acc_bias.ravel(),
#             gyro_bias.ravel()
#         ])
    
#     @property
#     def last_t_us(self):
#         return self._last_update_time
    
#     def get_evolving_state(self):
#         """MSCKF-compatible state getter for logging"""
#         # Extract state components
#         pos = self.ukf.x[0:3].reshape(3, 1)       # Position (3x1)
#         vel = self.ukf.x[3:6].reshape(3, 1)       # Velocity (3x1)
#         quat = self.ukf.x[6:10]                   # Quaternion (4,)
#         R_wi = Rotation.from_quat(quat).as_matrix()  # Rotation matrix (3x3)
#         ba = self.ukf.x[10:13].reshape(3, 1)      # Accel bias (3x1)
#         bg = self.ukf.x[13:16].reshape(3, 1)      # Gyro bias (3x1)
        
#         return R_wi, vel, pos, ba, bg
    
    

#     def measurement_model(self, x):
#         # Include pseudo-measurements for Y/Z
#         return np.concatenate(
#             x[3:6],  # Velocity
#             [x[2]]
#             # [x[1], x[2]]  # Y/Z position
#         )
    
#     def update(self, measurement, meas_cov):
#         # Modified measurement matrix
#         H = np.zeros((5, 16))
#         H[0,3] = H[1,4] = H[2,5] = 1  # Velocity
#         meas_cov = np.diag([0.1, 0.1, 0.05])  # Z variance = 0.05 (higher confidence)
#         self._last_update_time = time.time_us()
#         super().update(measurement, meas_cov)
#         self.ukf.update(measurement, meas_cov, H=H)
#         self._normalize_quaternion()
#         self._post_update_checks()

    # def _normalize_quaternion(self):
        
    #     # rpy = Rotation.from_quat(q).as_euler('xyz')
        
    #     # # Apply pitch/roll constraints
    #     # rpy[0] = np.clip(rpy[0], -0.2, 0.2)  # Roll
    #     # rpy[1] = np.clip(rpy[1], -0.2, 0.2)  # Pitch
        
    #     # new_q = Rotation.from_euler('xyz', rpy).as_quat()
    #     # self.ukf.x[6:10] = new_q / np.linalg.norm(new_q)
    #     q = self.ukf.x[6:10]
    #     q /= np.linalg.norm(q)
    #     self.ukf.x[6:10] = q

#     def _post_update_checks(self):
#         self.ukf.x[2] *= 0.99  # Z

#     def initialize_with_state(self, t_us, R, v, p, ba, bg):
#         """Initialize filter with specific state values"""
#         quat = Rotation.from_matrix(R).as_quat()
#         self.ukf.x = np.concatenate([
#             p.flatten(), v.flatten(), quat, ba.flatten(), bg.flatten()
#         ])
#         self.initialized = True
#         # self.last_t_us = t_us

#     def predict(self, acc, gyro, dt):
#         """Store flattened IMU measurements"""
#         self.current_imu = {
#             'acc': np.asarray(acc).flatten(),
#             'gyro': np.asarray(gyro).flatten()
#         }
#         self.ukf.predict(dt=dt)
#         self._normalize_quaternion()

    # def initialize_filter(self):
    #     # Axis-aware covariance initialization
    #     self.ukf.P = np.diag([
    #         # Position (X, Y, Z)
    #         0.1, 2.0, 1.5,  
    #         # Velocity (X, Y, Z)
    #         0.2, 3.0, 2.0,  
    #         # Orientation (qw, qx, qy, qz)
    #         0.01, 0.1, 0.1, 0.01,
    #         # Accel bias (X, Y, Z)
    #         0.05, 0.1, 0.1,
    #         # Gyro bias (X, Y, Z)
    #         0.005, 0.01, 0.01
    #     ])**2

    #     # Dynamic process noise
    #     self.ukf.Q = np.diag([
    #         0.1, 0.8, 0.6,   # Position
    #         0.2, 1.2, 0.8,   # Velocity
    #         0.01, 0.2, 0.2, 0.01,  # Orientation
    #         0.02, 0.1, 0.1,  # Accel bias
    #         0.001, 0.01, 0.01  # Gyro bias
    #     ]) * self.dt


    # def initialize_filter(self):
    #     # Use tuning parameters for process noise
    #     dt = self.dt
    #     # Extract parameters
    #     pos_sigma = np.array([0.1, 1.8, 1.2])
    #     vel_var = np.array([0.2**2, 0.3**2, 5.0**2])  # Directly compute velocity variances
    #     attitude_sigma = np.array([0.01, 0.15, 0.12, 0.01])
    #     sigma_nba = np.array([0.00008, 0.0002, 0.00015])
    #     sigma_nbg = np.array([0.000008, 0.00002, 0.000015])
    #     nba = sigma_nba
    #     nbg = sigma_nbg
        
    #     # Build covariance matrix
    #     diag_elements = np.concatenate([
    #         pos_sigma**2,          # 3 elements
    #         vel_var,                # 3 elements (already squared)
    #         attitude_sigma**2,     # 4 elements
    #         (0.5 * sigma_nba)**2,  # 3 elements
    #         (0.5 * sigma_nbg)**2   # 3 elements
    #     ])

    #     self.ukf.P = np.diag(diag_elements.astype(float))

    #     sigma_na=np.array([0.008, 0.025, 0.02])  # X/Y/Z accel
    #     sigma_ng=np.array([0.0008, 0.002, 0.0015])
        

    #     self.ukf.Q = np.diag([
    #         # Position noise (derived from acceleration noise)
    #         *(sigma_na**2 * dt**3), 
    #         # Velocity noise (derived from acceleration noise)
    #         *(sigma_na**2 * dt),  
    #         # Orientation noise (derived from gyro noise)
    #         *(sigma_ng**2 * dt),  
    #         0.0,  # Quaternion has 4 elements; last term padded
    #         # Accel bias random walk
    #         *(nba**2 * dt),  
    #         # Gyro bias random walk
    #         *(nbg**2 * dt)   
    #     ])[:self.state_dim, :self.state_dim]  # Trim to match state_dim

    #     # In initialize_filter()
    #     self.ukf.Q[10:13, 10:13] = np.diag(sigma_nba**2 * dt)
    #     self.ukf.Q[13:16, 13:16] = np.diag(sigma_nbg**2 * dt)

# class FilterRunner:
#     def __init__(self, model_path, model_param_path, update_freq, filter_tuning, imu_calib_dic=None):
#         # Load network configuration
#         with open(model_param_path) as f:
#             config = json.load(f)
        
#         self.imu_freq_net = config["sampling_freq"]
#         self.window_time = config["window_time"]
#         self.net_input_size = int(self.window_time * self.imu_freq_net)
#         self.update_freq = update_freq
#         self.dt_update_us = int(1e6 / update_freq)
        
#         # IMU calibration
#         self.icalib = ImuCalib()
#         if imu_calib_dic:
#             self.icalib.from_dic(imu_calib_dic)
            
#         # Initialize UKF
#         self.filter = UKFInertialOdometry(filter_tuning)
#         self.inputs_buffer = NetInputBuffer()
        
#         # State tracking
#         self.last_t_us = None
#         self.next_update_t_us = None
#         self.last_acc = None
#         self.last_gyr = None
#         self.last_thrust = None

#     def _process_network_update(self, t_us):
#         # Modified measurement handling
#         net_gyr_w, net_fn_w, net_t_s = self._get_inputs_samples_for_network()
#         meas_velocity, meas_cov = self.meas_source.get_displacement_measurement(net_t_s, net_gyr_w, net_fn_w)
        
#         # Add pseudo-measurements (zeros with large covariance)
#         measurement = np.concatenate([
#             meas_velocity,
#             [0, 0]  # Y/Z position pseudo-measurements
#         ])
#         meas_cov = np.diag([
#             *np.diag(meas_cov),
#             5.0,  # Y position covariance
#             5.0    # Z position covariance
#         ])
        
#         self.filter.update(measurement, meas_cov)

#     def on_imu_measurement(self, t_us, gyr_raw, acc_raw, thrust=None):
#         """Process IMU measurement"""
#         if not self.filter.initialized:
#             return self._initialize_filter(t_us, gyr_raw, acc_raw)
            
#         # Calibrate IMU
#         acc, gyr = self._calibrate_imu(acc_raw, gyr_raw)
        
#         # Calculate time step
#         dt = from_usec_to_sec(t_us - self.filter.last_t_us)
        
#         # UKF prediction
#         self.filter.predict(acc, gyr, dt)
        
#         # Update tracking variables
#         self.last_acc = acc
#         self.last_gyr = gyr
#         self.last_thrust = thrust
#         self.filter.last_t_us = t_us
        
#         return True
    

#     def _initialize_filter(self, t_us, gyr_raw, acc_raw):
#         """Initialize UKF state"""
#         if self.icalib:
#             init_ba = self.icalib.accelBias
#             init_bg = self.icalib.gyroBias
#             acc, gyr = self.icalib.calibrate(acc_raw, gyr_raw)
#         else:
#             init_ba = np.zeros((3,1))
#             init_bg = np.zeros((3,1))
#             acc, gyr = acc_raw, gyr_raw

#         self.filter.initialize_with_state(
#             t_us=t_us,
#             R=np.eye(3),
#             v=np.zeros((3,1)),
#             p=np.zeros((3,1)),
#             ba=init_ba,
#             bg=init_bg
#         )
#         return False

#     def _calibrate_imu(self, acc_raw, gyr_raw):
#         """Ensure 1D array output with explicit flattening"""
#         if self.icalib:
#             acc, gyr = self.icalib.calibrate_raw(acc_raw, gyr_raw)
#         else:
#             acc, gyr = acc_raw, gyr_raw
            
#         return np.asarray(acc).flatten(), np.asarray(gyr).flatten()

#     def predict(self, acc, gyro, dt):
#         """Store flattened IMU measurements"""
#         self.current_imu = {
#             'acc': np.asarray(acc).flatten(),
#             'gyro': np.asarray(gyro).flatten()
#         }
#         self.ukf.predict(dt=dt)
#         self._normalize_quaternion()



#-------------------------------------------------------------------------------------------




# """
# This file is part of Learned Inertial Model Odometry.
# Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
# (Robotics and Perception Group, University of Zurich, Switzerland).
# This file is subject to the terms and conditions defined in the file
# 'LICENSE', which is part of this source code package.
# """

# import os
# import numpy as np
# import progressbar
# from scipy.interpolate import interp1d
# from scipy.spatial.transform import Rotation, Slerp
# from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# from filter.python.src.data_io import DataIO
# from filter.python.src.utils.dotdict import dotdict
# from filter.python.src.utils.logging import logging
# from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec

# class UKFInertialOdometry:
#     def __init__(self, filter_tuning):
#         # State dimensions [pos(3), vel(3), quat(4), acc_bias(3), gyro_bias(3)]
#         self.state_dim = 16
#         self.meas_dim = 3  # Network provides velocity measurements
#         self.filter_tuning = filter_tuning
#         self.dt = 1.0/filter_tuning.get('imu_freq', 100)  # Default 200Hz
        
#         # Sigma points
#         self.sigma_points = MerweScaledSigmaPoints(
#             n=self.state_dim,
#             alpha=0.25,
#             beta=2.0,
#             kappa=0.0
#         )
        
#         # UKF instance
#         self.ukf = UnscentedKalmanFilter(
#             dim_x=self.state_dim,
#             dim_z=self.meas_dim,
#             dt=self.dt,
#             fx=self.state_transition,
#             hx=self.measurement_model,
#             points=self.sigma_points
#         )
        
#         # Initialize state and covariance
#         self.initialize_filter()
#         self.g = np.array([0, 0, 0])
#         self.current_imu = {'acc': None, 'gyro': None}
#         self.initialized = False
#         self.last_t_us = None

#     def initialize_filter(self):
#         """Initialize state and covariance"""
#         self.ukf.x = np.zeros(self.state_dim)
#         self.ukf.x[6:10] = [1, 0, 0, 0]  # Identity quaternion
        
#         # Initial covariance
#         self.ukf.P = np.diag([
#             *[self.filter_tuning.init_pos_sigma**2]*3,
#             *[self.filter_tuning.init_vel_sigma**2]*3,
#             *[self.filter_tuning.init_attitude_sigma**2]*4,
#             *[self.filter_tuning.init_ba_sigma**2]*3,
#             *[self.filter_tuning.init_bg_sigma**2]*3
#         ])
        
#         # Process noise
#         self.ukf.Q = np.diag([
#             *[self.filter_tuning.sigma_na**2 * self.dt**2]*3,
#             *[self.filter_tuning.sigma_na**2 * self.dt]*3,
#             *[self.filter_tuning.sigma_ng**2 * self.dt]*4,
#             *[self.filter_tuning.sigma_nba**2 * self.dt]*3,
#             *[self.filter_tuning.sigma_nbg**2 * self.dt]*3
#         ])

#     def state_transition(self, x, dt):
#         """UKF state transition function"""
#         pos = x[0:3].flatten()     # Ensure 1D (3,)
#         vel = x[3:6].flatten()     # Ensure 1D (3,)
#         quat = x[6:10].flatten()   # Ensure 1D (4,)
#         acc_bias = x[10:13].flatten()
#         gyro_bias = x[13:16].flatten()
        
#         # IMU measurements with bias correction (ensure 1D)
#         acc = np.asarray(self.current_imu['acc']).flatten() - acc_bias
#         gyro = np.asarray(self.current_imu['gyro']).flatten() - gyro_bias
        
#         # Orientation integration
#         rot = Rotation.from_quat(quat)
#         d_rot = Rotation.from_rotvec(gyro * dt)
#         new_rot = rot * d_rot
#         new_quat = new_rot.as_quat().ravel()
        
#         # Acceleration in world frame
#         acc_world = (new_rot.apply(acc) + self.g).flatten()

#         # Position and velocity integration
#         new_pos = (pos + vel * dt + 0.5 * acc_world * dt**2).flatten()
#         new_vel = (vel + acc_world * dt).flatten()
        
        
#         return np.concatenate([
#             new_pos,    # Flatten to (3,)
#             new_vel,    # Flatten to (3,)
#             new_quat,   # Flatten to (4,)
#             acc_bias,   # Flatten to (3,)
#             gyro_bias   # Flatten to (3,)
#         ])

#     def measurement_model(self, x):
#         """Velocity measurement model"""
#         return x[3:6]

#     def initialize_with_state(self, t_us, R, v, p, ba, bg):
#         """Initialize filter with specific state values"""
#         quat = Rotation.from_matrix(R).as_quat()
#         self.ukf.x = np.concatenate([
#             p.flatten(), v.flatten(), quat, ba.flatten(), bg.flatten()
#         ])
#         self.initialized = True
#         self.last_t_us = t_us

#     def get_evolving_state(self):
#         """Get current state in MSCKF-compatible format"""
#         R = Rotation.from_quat(self.ukf.x[6:10]).as_matrix()
#         return (
#             R,
#             self.ukf.x[3:6].reshape(3, 1),
#             self.ukf.x[0:3].reshape(3, 1),
#             self.ukf.x[10:13].reshape(3, 1),
#             self.ukf.x[13:16].reshape(3, 1)
#         )

#     def predict(self, acc, gyro, dt):
#         """UKF prediction step"""
#         self.current_imu = {'acc': acc, 'gyro': gyro}
#         self.ukf.predict(dt=dt)
#         self._normalize_quaternion()

#     def update(self, measurement, meas_cov):
#         """UKF update step"""
#         self.ukf.R = meas_cov
#         self.ukf.update(measurement)
#         self._normalize_quaternion()

#     def _normalize_quaternion(self):
#         """Maintain quaternion unit norm"""
#         q = self.ukf.x[6:10]
#         self.ukf.x[6:10] = q / np.linalg.norm(q)

# class FilterRunner:
#     def __init__(self, model_path, model_param_path, update_freq, filter_tuning, imu_calib_dic=None):
#         # Load network configuration
#         with open(model_param_path) as f:
#             config = json.load(f)
        
#         self.imu_freq_net = config["sampling_freq"]
#         self.window_time = config["window_time"]
#         self.net_input_size = int(self.window_time * self.imu_freq_net)
#         self.update_freq = update_freq
#         self.dt_update_us = int(1e6 / update_freq)
        
#         # IMU calibration
#         self.icalib = ImuCalib()
#         if imu_calib_dic:
#             self.icalib.from_dic(imu_calib_dic)
            
#         # Initialize UKF
#         self.filter = UKFInertialOdometry(filter_tuning)
#         self.inputs_buffer = NetInputBuffer()
        
#         # State tracking
#         self.last_t_us = None
#         self.next_update_t_us = None
#         self.last_acc = None
#         self.last_gyr = None
#         self.last_thrust = None

#     def on_imu_measurement(self, t_us, gyr_raw, acc_raw, thrust=None):
#         """Process IMU measurement"""
#         if not self.filter.initialized:
#             return self._initialize_filter(t_us, gyr_raw, acc_raw)
            
#         # Calibrate IMU
#         acc, gyr = self._calibrate_imu(acc_raw, gyr_raw)
        
#         # Calculate time step
#         dt = from_usec_to_sec(t_us - self.filter.last_t_us)
        
#         # UKF prediction
#         self.filter.predict(acc, gyr, dt)
        
#         # Update tracking variables
#         self.last_acc = acc
#         self.last_gyr = gyr
#         self.last_thrust = thrust
#         self.filter.last_t_us = t_us
        
#         return True

#     def _initialize_filter(self, t_us, gyr_raw, acc_raw):
#         """Initialize UKF state"""
#         if self.icalib:
#             init_ba = self.icalib.accelBias
#             init_bg = self.icalib.gyroBias
#             acc, gyr = self.icalib.calibrate(acc_raw, gyr_raw)
#         else:
#             init_ba = np.zeros((3,1))
#             init_bg = np.zeros((3,1))
#             acc, gyr = acc_raw, gyr_raw

#         self.filter.initialize_with_state(
#             t_us=t_us,
#             R=np.eye(3),
#             v=np.zeros((3,1)),
#             p=np.zeros((3,1)),
#             ba=init_ba,
#             bg=init_bg
#         )
#         return False

#     def _calibrate_imu(self, acc_raw, gyr_raw):
#         """Apply IMU calibration"""
#         if self.icalib:
#             return self.icalib.calibrate_raw(acc_raw, gyr_raw)
#         return acc_raw.ravel(), gyr_raw.ravel()

# # Rest of the FilterManager class remains the same as in the original question
# # but now using the UKF implementation above