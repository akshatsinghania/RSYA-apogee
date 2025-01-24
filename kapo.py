import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty):
        """
        Initializes the Kalman Filter parameters.
        :param initial_state: Initial state estimate (position, velocity, etc.)
        :param initial_uncertainty: Initial uncertainty estimate (covariance)
        :param process_uncertainty: Process noise covariance (uncertainty in the model)
        :param measurement_uncertainty: Measurement noise covariance (uncertainty in the sensor)
        """
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_uncertainty = process_uncertainty
        self.measurement_uncertainty = measurement_uncertainty

    def predict(self, control_input=0):
        """
        Predict the next state of the system using the Kalman filter prediction step.
        :param control_input: The control input (acceleration or other input to the system).
        """
        # State transition matrix (A) for a simple model with position and velocity
        A = np.array([[1, 1], [0, 1]])  # Position and velocity model
        # Control input matrix (B)
        B = np.array([0.5, 1])  # How acceleration impacts position and velocity

        # Predicted state estimate
        self.state = np.dot(A, self.state) + B * control_input
        # Predicted uncertainty
        self.uncertainty = np.dot(A, np.dot(self.uncertainty, A.T)) + self.process_uncertainty

    def update(self, measurement):
        """
        Update the Kalman filter state using the measurement update step.
        :param measurement: The measurement (altimeter data).
        """
        # Measurement matrix (H)
        H = np.array([1, 0])  # We are only measuring position (altitude)

        # Kalman gain calculation
        S = np.dot(H, np.dot(self.uncertainty, H.T)) + self.measurement_uncertainty
        K = np.dot(self.uncertainty, H.T) / S

        # Update state estimate
        self.state = self.state + K * (measurement - np.dot(H, self.state))
        # Update uncertainty
        self.uncertainty = self.uncertainty - np.dot(K, H) * self.uncertainty

    def get_state(self):
        """
        Returns the current state estimate (position, velocity).
        """
        return self.state

# Load data from CSV

# Skip comment lines starting with '#' when reading the CSV
data = pd.read_csv('altdata.csv', comment='#')

# Display the column names to check what they are
print(data.columns)

# Assuming the CSV has columns 'time' and 'altitude' (adjust as necessary)
time_steps = data['time'].values  # Time steps from the data
measurements = data['altitude'].values  # Altimeter data (altitude)

# Kalman Filter Parameters
initial_state = np.array([0, 0])  # [position, velocity]
initial_uncertainty = np.array([[1000, 0], [0, 1000]])  # Initial uncertainty
process_uncertainty = np.array([[1, 0], [0, 1]])  # Process noise (uncertainty in model)
measurement_uncertainty = 1  # Measurement noise (uncertainty in sensor)

# Create Kalman Filter instance
kf = KalmanFilter(initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty)

# Apply Kalman Filter on the measurements
filtered_positions = []
filtered_velocities = []

for measurement in measurements:
    kf.predict()  # Predict next state
    kf.update(measurement)  # Update state with new measurement
    state = kf.get_state()  # Get the current state estimate
    filtered_positions.append(state[0])  # Position estimate
    filtered_velocities.append(state[1])  # Velocity estimate

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot measurements
plt.subplot(2, 1, 1)
plt.plot(time_steps, measurements, label='Measured Altitude', color='red')
plt.plot(time_steps, filtered_positions, label='Filtered Altitude (Kalman)', color='blue')
plt.title("Altitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()

# Plot velocity estimate
plt.subplot(2, 1, 2)
plt.plot(time_steps, filtered_velocities, label='Filtered Velocity (Kalman)', color='green')
plt.title("Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()
