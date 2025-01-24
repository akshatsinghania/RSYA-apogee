import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty):
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_uncertainty = process_uncertainty
        self.measurement_uncertainty = measurement_uncertainty

    def predict(self, control_input=0):
        A = np.array([[1, 1], [0, 1]])
        B = np.array([0.5, 1])
        self.state = np.dot(A, self.state) + B * control_input
        self.uncertainty = np.dot(A, np.dot(self.uncertainty, A.T)) + self.process_uncertainty

    def update(self, measurement):
        H = np.array([1, 0])
        S = np.dot(H, np.dot(self.uncertainty, H.T)) + self.measurement_uncertainty
        K = np.dot(self.uncertainty, H.T) / S
        self.state = self.state + K * (measurement - np.dot(H, self.state))
        self.uncertainty = self.uncertainty - np.dot(K, H) * self.uncertainty

    def get_state(self):
        return self.state

# Load data from CSV (skipping comment lines)
data = pd.read_csv('altdata.csv', comment='#')

# Display the column names to verify
print(data.columns)

# Assuming the columns are 'Time (s)' and 'Altitude (m)' (adjust if needed)
time_steps = data['time'].values
measurements = data['altitude'].values

# Kalman Filter parameters
initial_state = np.array([0, 0])  # [position, velocity]
initial_uncertainty = np.array([[1000, 0], [0, 1000]])
process_uncertainty = np.array([[1, 0], [0, 1]])
measurement_uncertainty = 1

# Create Kalman Filter instance
kf = KalmanFilter(initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty)

# Apply Kalman Filter on the measurements
filtered_positions = []
filtered_velocities = []

# Track events
liftoff_time = None
apogee_time = None

# To identify when the altitude stops increasing (apogee)
max_altitude = -np.inf
for i, measurement in enumerate(measurements):
    kf.predict()
    kf.update(measurement)
    state = kf.get_state()
    
    filtered_positions.append(state[0])
    filtered_velocities.append(state[1])
    
    # Detect liftoff: First time altitude > 0
    if liftoff_time is None and state[0] > 0:
        liftoff_time = time_steps[i]

    # Detect apogee: Maximum altitude point
    if state[0] > max_altitude:
        max_altitude = state[0]
        apogee_time = time_steps[i]

# Output detected events
print(f"Liftoff Time: {liftoff_time} seconds")
print(f"Apogee Time: {apogee_time} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot measured vs filtered altitude
plt.subplot(2, 1, 1)
plt.plot(time_steps, measurements, label='Measured Altitude', color='red')
plt.plot(time_steps, filtered_positions, label='Filtered Altitude (Kalman)', color='blue')
plt.title("Altitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()

# Plot filtered velocity
plt.subplot(2, 1, 2)
plt.plot(time_steps, filtered_velocities, label='Filtered Velocity (Kalman)', color='green')
plt.title("Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()
