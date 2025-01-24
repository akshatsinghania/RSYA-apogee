import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Load the data
data = pd.read_csv('altdata.csv', comment='#')
time_steps = data['time'].values
measurements = data['altitude'].values

# Initialize the Kalman Filter
initial_state = np.array([0, 0])  
initial_uncertainty = np.array([[1000, 0], [0, 1000]])
process_uncertainty = np.array([[1, 0], [0, 1]])
measurement_uncertainty = 1

kf = KalmanFilter(initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty)

# Prepare lists for storing results
filtered_positions = []
filtered_velocities = []
liftoff_time = None
apogee_time = None
max_altitude = -np.inf

# Set up the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Initialize the plots
line_measured, = ax1.plot([], [], label='Measured Altitude', color='red')
line_filtered_alt, = ax1.plot([], [], label='Filtered Altitude (Kalman)', color='blue')
liftoff_marker, = ax1.plot([], [], 'o', color='orange', label='Liftoff')
apogee_marker, = ax1.plot([], [], 'o', color='purple', label='Apogee')

line_velocity, = ax2.plot([], [], label='Filtered Velocity (Kalman)', color='green')

# Set plot limits and labels
ax1.set_title("Altitude Over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Altitude (m)")
ax1.legend()

ax2.set_title("Velocity Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.legend()

def init():
    line_measured.set_data([], [])
    line_filtered_alt.set_data([], [])
    liftoff_marker.set_data([], [])
    apogee_marker.set_data([], [])
    line_velocity.set_data([], [])
    return line_measured, line_filtered_alt, liftoff_marker, apogee_marker, line_velocity

def update(frame):
    global liftoff_time, apogee_time, max_altitude

    # Process the next data point
    measurement = measurements[frame]
    kf.predict()
    kf.update(measurement)
    state = kf.get_state()

    # Update the lists
    filtered_positions.append(state[0])
    filtered_velocities.append(state[1])

    # Log current state for debugging
    print(f"Frame: {frame}, Time: {time_steps[frame]}, Measured: {measurement}, Filtered Altitude: {state[0]}, Velocity: {state[1]}")

    # Check for liftoff time
    if liftoff_time is None and state[0] > 0:
        liftoff_time = time_steps[frame]

    # Check for apogee
    if state[0] > max_altitude:
        max_altitude = state[0]
        apogee_time = time_steps[frame]

    # Update plot data
    line_measured.set_data(time_steps[:frame+1], measurements[:frame+1])
    line_filtered_alt.set_data(time_steps[:frame+1], filtered_positions)
    liftoff_marker.set_data([liftoff_time], [filtered_positions[time_steps.tolist().index(liftoff_time)]])
    apogee_marker.set_data([apogee_time], [filtered_positions[time_steps.tolist().index(apogee_time)]])
    line_velocity.set_data(time_steps[:frame+1], filtered_velocities)

    # Update axes limits dynamically
    ax1.set_xlim(0, time_steps[:frame+1].max() + 1)
    ax1.set_ylim(0, max(filtered_positions) + 1)
    ax2.set_xlim(0, time_steps[:frame+1].max() + 1)
    ax2.set_ylim(min(filtered_velocities) - 1, max(filtered_velocities) + 1)

    return line_measured, line_filtered_alt, liftoff_marker, apogee_marker, line_velocity

# Define the simulation speed: simulation seconds per real-life second
simulation_speed = 100000  # For example, 10 simulation seconds per real-life second

# Calculate interval based on simulation_speed
# interval = (real-life milliseconds per frame) = (simulation seconds per frame / simulation_speed) * 1000
simulation_time_per_frame = time_steps[1] - time_steps[0]  # Time difference between consecutive steps
interval = (simulation_time_per_frame / simulation_speed) * 1000
print(interval)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_steps), init_func=init, blit=True, interval=interval)


plt.tight_layout()
plt.show()
