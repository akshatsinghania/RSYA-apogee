import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, measurement_variance, model_variance, dt):
        """
        Initializes the Kalman filter.

        Args:
            initial_state (list): The initial state estimate [position, velocity, acceleration].
            initial_covariance (list): The initial error covariance matrix.
            measurement_variance (float): Variance of the measurement noise.
            model_variance (float): Variance of the model noise.
            dt (float): Time between samples.
        """
        self.state = np.array(initial_state, dtype=float)
        self.covariance = np.array(initial_covariance, dtype=float)
        self.measurement_variance = measurement_variance
        self.model_variance = model_variance
        self.dt = dt
        self.phi = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]], dtype=float)
        self.phi_transpose = self.phi.T
        # Precomputed kalman gain
        self.gain = np.array([0.010317, 0.010666, 0.004522], dtype=float)


    def predict(self):
      """
      Predicts the next state using the system model.
      """
      self.predicted_state = np.dot(self.phi, self.state)
      # Propagate state covariance. Simplified version of the matrix math
      term = np.dot(self.phi, self.covariance)
      self.predicted_covariance = np.dot(term, self.phi_transpose)
      self.predicted_covariance += self.model_variance

    def update(self, measurement):
        """
        Updates the state estimate with a new measurement.
        Args:
            measurement (float): The current pressure measurement.

        Returns:
          float: The estimated velocity
        """
        # Simplified update equations using pre-computed kalman gain
        innovation = measurement - self.predicted_state
        self.state = self.predicted_state + self.gain * innovation
        self.state[1] = self.predicted_state[1] + self.gain[1] * innovation
        self.state[2] = self.predicted_state[2] + self.gain[2] * innovation
        # Simplified state covariance update
        self.covariance = self.predicted_covariance * (1 - self.gain)
        self.covariance[1] = self.predicted_covariance[1] * (1 - self.gain)
        self.covariance[2] = self.predicted_covariance[2] * (1 - self.gain)
        self.covariance[1] = self.predicted_covariance[1] - self.gain[1] * self.predicted_covariance
        self.covariance[1, 1] = self.predicted_covariance[1, 1] - self.gain[1] * self.predicted_covariance[1]
        self.covariance[1, 2] = self.predicted_covariance[1, 2] - self.gain[1] * self.predicted_covariance[2]
        self.covariance[2] = self.predicted_covariance[2] - self.gain[2] * self.predicted_covariance
        self.covariance[1, 2] = self.predicted_covariance[1, 2] - self.gain[2] * self.predicted_covariance[1]
        self.covariance[2, 2] = self.predicted_covariance[2, 2] - self.gain[2] * self.predicted_covariance[2]
        return self.state[1]

def detect_apogee(kalman_filter, measurements):
  """
  Detects apogee based on the velocity estimated by the Kalman filter

  Args:
      kalman_filter (KalmanFilter): Initialized Kalman filter object.
      measurements (list): List of pressure measurements.

  Returns:
      float: Time of apogee detection
  """
  last_velocity = 0
  for time, measurement in enumerate(measurements):
    kalman_filter.predict()
    current_velocity = kalman_filter.update(measurement)

    if last_velocity > 0 and current_velocity < 0:
      return time
    last_velocity = current_velocity
  return None

KL = KalmanFilter(
    initial_state=[0, 0, 0],
    initial_covariance=[[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]],  # Initial covariance matrix
    measurement_variance=0.44**2,  # Measurement variance
    model_variance=0.002**2,  # Model variance
    dt=0.005  # Time between samples if sensor is sampled at 200 SPS (1/200)
)

measurements = [1, 2,3,4,5,6,7,8,9,10,11,10,9,8,7,6,5,4,3,2,1,0]  # Example list of pressure measurements

apogee_time = detect_apogee(KL, measurements)
print(apogee_time)

