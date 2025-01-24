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
        self.uncertainty = self.uncertainty - np.outer(K, np.dot(H, self.uncertainty))
        
        return K  # Return Kalman Gain for visualization

    def get_state(self):
        return self.state

# Load the data
data = pd.read_csv('data1.csv', comment='#')
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
apogee_times = []
liftoff_time = None
apogee_time = None
max_altitude = -np.inf
stable_apogee_count = 0
stability_threshold = 10  # Number of frames to consider apogee stable

# Set up the figure for altitude plot and apogee time plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Initialize the plots
line_measured, = ax1.plot([], [], label='Measured Altitude', color='red')
line_filtered_alt, = ax1.plot([], [], label='Filtered Altitude (Kalman)', color='blue', linestyle=':')
liftoff_marker, = ax1.plot([], [], 'o', color='orange', label='Liftoff')
apogee_marker, = ax1.plot([], [], 'o', color='purple', label='Apogee')

line_apogee_time, = ax2.plot([], [], label='Apogee Time', color='green')

# Set plot limits and labels
ax1.set_title("Altitude Over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Altitude (m)")
ax1.legend()

ax2.set_title("Apogee Time Over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Apogee Time (s)")
ax2.legend()

def init():
    line_measured.set_data([], [])
    line_filtered_alt.set_data([], [])
    liftoff_marker.set_data([], [])
    apogee_marker.set_data([], [])
    line_apogee_time.set_data([], [])
    return line_measured, line_filtered_alt, liftoff_marker, apogee_marker, line_apogee_time

# Define the number of simulation steps per animation frame
steps_per_frame = 50  # Simulate 5 time steps for every frame

def update(frame):
    global liftoff_time, apogee_time, max_altitude, stable_apogee_count

    for _ in range(steps_per_frame):
        current_frame = frame * steps_per_frame + _
        if current_frame >= len(time_steps):  # Prevent going out of bounds
            break

        measurement = measurements[current_frame]
        kf.predict()
        kf.update(measurement)  # Update without storing Kalman Gain
        state = kf.get_state()

        filtered_positions.append(state[0])

        if liftoff_time is None and state[0] > 0:
            liftoff_time = time_steps[current_frame]

        if state[0] > max_altitude:
            max_altitude = state[0]
            apogee_time = time_steps[current_frame]
            stable_apogee_count = 0  # Reset stability count when a new apogee is found
        else:
            stable_apogee_count += 1

        apogee_times.append(apogee_time if apogee_time else 0)

    # Check if apogee time is stable
    if stable_apogee_count >= stability_threshold:
        print(f"Apogee detected at time: {apogee_time}")

    # Limit the length of the lists to match the time steps
    filtered_positions_trimmed = filtered_positions[:len(time_steps)]
    apogee_times_trimmed = apogee_times[:len(time_steps)]

    # Update plot data
    line_measured.set_data(time_steps[:len(filtered_positions_trimmed)], measurements[:len(filtered_positions_trimmed)])
    line_filtered_alt.set_data(time_steps[:len(filtered_positions_trimmed)], filtered_positions_trimmed)
    line_apogee_time.set_data(time_steps[:len(apogee_times_trimmed)], apogee_times_trimmed)

    # Update axis limits for altitude plot
    ax1.set_xlim(0, time_steps[:len(filtered_positions_trimmed)][-1] + 1)  # Dynamic time axis
    ax1.set_ylim(min(measurements[:len(filtered_positions_trimmed)]) - 10, max(filtered_positions_trimmed) + 10)  # Dynamic altitude

    # Update axis limits for apogee time plot
    ax2.set_xlim(0, time_steps[:len(apogee_times_trimmed)][-1] + 1)  # Dynamic time axis
    ax2.set_ylim(0, max(apogee_times_trimmed) + 10)  # Dynamic apogee time

    if liftoff_time:
        liftoff_index = np.where(time_steps == liftoff_time)[0][0]
        liftoff_marker.set_data([liftoff_time], [filtered_positions_trimmed[liftoff_index]])

    if apogee_time:
        apogee_index = np.where(time_steps == apogee_time)[0][0]
        apogee_marker.set_data([apogee_time], [filtered_positions_trimmed[apogee_index]])

    return line_measured, line_filtered_alt, liftoff_marker, apogee_marker, line_apogee_time

# Animation interval settings
simulation_speed = 1.0  # Adjust to speed up/slow down the simulation
interval = 100  # 100ms between frames

# Create the animation
ani = FuncAnimation(fig, update, frames=(len(time_steps) // steps_per_frame), init_func=init, blit=True, interval=interval)

plt.tight_layout()
plt.show()
