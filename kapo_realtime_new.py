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
        self.state_before_update = initial_state
        self.state_after_update = initial_state

    def predict(self, control_input=0):
        A = np.array([[1, 1], [0, 1]])
        B = np.array([0.5, 1])
        self.state_before_update = np.dot(A, self.state) + B * control_input
        self.uncertainty = np.dot(A, np.dot(self.uncertainty, A.T)) + self.process_uncertainty

    def update(self, measurement):
        H = np.array([1, 0])
        S = np.dot(H, np.dot(self.uncertainty, H.T)) + self.measurement_uncertainty
        K = np.dot(self.uncertainty, H.T) / S
        self.state_after_update = self.state_before_update + K * (measurement - np.dot(H, self.state_before_update))
        self.uncertainty = self.uncertainty - np.outer(K, np.dot(H, self.uncertainty))
        
        return K  # Return Kalman Gain for visualization

    def get_state(self):
        return self.state_after_update

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
filtered_velocities = []
kalman_gains = []  # For Kalman Gain visualization
uncertainty_values = []  # For Uncertainty visualization
innovation_values = []  # For Innovation visualization
covariance_values = []  # For Error Covariance visualization
process_noise_values = []  # For Process Noise visualization
measurement_noise_values = []  # For Measurement Noise visualization
raw_positions = []  # For raw data visualization
raw_velocities = []  # For raw data velocity visualization
state_update_rate = []  # For State Update Rate visualization
liftoff_time = None
apogee_time = None
max_altitude = -np.inf
# Set up the figure with additional subplots for uncertainty and Kalman Gain
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, figsize=(10, 15))

# Initialize the plots
line_measured, = ax1.plot([], [], label='Measured Altitude', color='red')
line_filtered_alt, = ax1.plot([], [], label='Filtered Altitude (Kalman)', color='blue', linestyle=':')
line_raw_alt, = ax1.plot([], [], label='Raw Altitude', color='gray', linestyle='--')

line_velocity, = ax2.plot([], [], label='Filtered Velocity (Kalman)', color='green')
line_raw_velocity, = ax2.plot([], [], label='Raw Velocity', color='orange', linestyle='--')

line_uncertainty, = ax3.plot([], [], label='Uncertainty in Altitude', color='purple')
line_kalman_gain, = ax4.plot([], [], label='Kalman Gain', color='orange')

line_innovation, = ax5.plot([], [], label='Innovation (Measurement Residual)', color='cyan')
line_covariance, = ax6.plot([], [], label='Error Covariance', color='brown')

line_process_noise, = ax7.plot([], [], label='Process Noise', color='magenta')
line_measurement_noise, = ax8.plot([], [], label='Measurement Noise', color='yellow')

line_state_update_rate, = ax9.plot([], [], label='State Update Rate', color='pink')

# Precompute data for all plots
def preprocess_data():
    for i, measurement in enumerate(measurements):
        kf.predict()
        K = kf.update(measurement)  # Get Kalman Gain
        state = kf.get_state()

        filtered_positions.append(state[0])
        filtered_velocities.append(state[1])
        kalman_gains.append(K[1])  # Store Kalman Gain
        uncertainty_values.append(kf.uncertainty[0, 0])  # Store Uncertainty in Altitude
        innovation = measurement - np.dot(np.array([1, 0]), state)  # Calculate Innovation
        innovation_values.append(innovation)
        covariance_values.append(kf.uncertainty[0, 0])  # Store Error Covariance
        process_noise_values.append(kf.process_uncertainty[0, 0])  # Process Noise
        measurement_noise_values.append(kf.measurement_uncertainty)  # Measurement Noise
        raw_positions.append(measurement)
        
        # Estimate velocity using np.gradient
        if i >= 2:
            raw_velocity = np.gradient(measurements[:i + 1])[-1]
        else:
            raw_velocity = 0
        raw_velocities.append(raw_velocity)

        if len(filtered_positions) > 1:
            state_update_rate.append(abs(filtered_positions[-1] - filtered_positions[-2]))

        # Track liftoff and apogee
        if liftoff_time is None and state[0] > 0:
            liftoff_time = time_steps[i]
        if state[0] > max_altitude:
            max_altitude = state[0]
            apogee_time = time_steps[i]

# Call preprocessing
preprocess_data()

# Adjust compute_axis_limits to use precomputed data
def compute_axis_limits():
    min_time = np.min(time_steps)
    max_time = np.max(time_steps)
    ax1.set_xlim(min_time, max_time)
    ax1.set_ylim(np.min(filtered_positions), np.max(filtered_positions))

    ax2.set_xlim(min_time, max_time)
    ax2.set_ylim(np.min(filtered_velocities), np.max(filtered_velocities))

    ax3.set_xlim(min_time, max_time)
    ax3.set_ylim(0, np.max(uncertainty_values))

    ax4.set_xlim(min_time, max_time)
    ax4.set_ylim(0, np.max(kalman_gains))

    ax5.set_xlim(min_time, max_time)
    ax5.set_ylim(np.min(innovation_values), np.max(innovation_values))

    ax6.set_xlim(min_time, max_time)
    ax6.set_ylim(0, np.max(covariance_values))

    ax7.set_xlim(min_time, max_time)
    ax7.set_ylim(0, np.max(process_noise_values))

    ax8.set_xlim(min_time, max_time)
    ax8.set_ylim(0, np.max(measurement_noise_values))

    ax9.set_xlim(min_time, max_time)
    ax9.set_ylim(0, np.max(state_update_rate))

# Initialize axes
compute_axis_limits()


def init():
    line_measured.set_data([], [])
    line_filtered_alt.set_data([], [])
    line_raw_alt.set_data([], [])
    line_velocity.set_data([], [])
    line_raw_velocity.set_data([], [])
    line_uncertainty.set_data([], [])
    line_kalman_gain.set_data([], [])
    line_innovation.set_data([], [])
    line_covariance.set_data([], [])
    line_process_noise.set_data([], [])
    line_measurement_noise.set_data([], [])
    line_state_update_rate.set_data([], [])
    return line_measured, line_filtered_alt, line_raw_alt, line_velocity, line_raw_velocity, line_uncertainty, line_kalman_gain, line_innovation, line_covariance, line_process_noise, line_measurement_noise, line_state_update_rate

# Define the number of simulation steps per animation frame
steps_per_frame = 5  # Simulate 5 time steps for every frame

def update(frame):
    global liftoff_time, apogee_time, max_altitude

    for _ in range(steps_per_frame):
        current_frame = frame * steps_per_frame + _
        if current_frame >= len(time_steps):  # Prevent going out of bounds
            break

        measurement = measurements[current_frame]
        kf.predict()
        K = kf.update(measurement)  # Get Kalman Gain
        state = kf.get_state()

        filtered_positions.append(state[0])
        filtered_velocities.append(state[1])
        kalman_gains.append(K[1])  # Store Kalman Gain for visualization
        uncertainty_values.append(kf.uncertainty[0, 0])  # Store Uncertainty in Altitude
        innovation = measurement - np.dot(np.array([1, 0]), state)  # Calculate Innovation
        innovation_values.append(innovation)

        covariance_values.append(kf.uncertainty[0, 0])  # Store Error Covariance
        process_noise_values.append(kf.process_uncertainty[0, 0])  # Process Noise
        measurement_noise_values.append(kf.measurement_uncertainty)  # Measurement Noise

        raw_positions.append(measurements[current_frame])
            # Estimate velocity using np.gradient, but only if there are enough data points
        if current_frame >= 2:  # Make sure there are enough data points
            raw_velocity = np.gradient(measurements[:current_frame + 1])[-1]  # Take only the last velocity value
        else:
            raw_velocity = 0  # Default value for the first few points where gradient can't be computed
        raw_velocities.append(raw_velocity)

        # Track state updates (this is just a simple difference measure)
        if len(filtered_positions) > 1:
            state_update_rate.append(abs(filtered_positions[-1] - filtered_positions[-2]))

        if liftoff_time is None and state[0] > 0:
            liftoff_time = time_steps[current_frame]

        if state[0] > max_altitude:
            max_altitude = state[0]
            apogee_time = time_steps[current_frame]

    # Trim lists to match the number of time steps
    filtered_positions_trimmed = filtered_positions[:len(time_steps)]
    filtered_velocities_trimmed = filtered_velocities[:len(time_steps)]
    kalman_gains_trimmed = kalman_gains[:len(time_steps)]
    uncertainty_values_trimmed = uncertainty_values[:len(time_steps)]
    innovation_values_trimmed = innovation_values[:len(time_steps)]
    covariance_values_trimmed = covariance_values[:len(time_steps)]
    process_noise_values_trimmed = process_noise_values[:len(time_steps)]
    measurement_noise_values_trimmed = measurement_noise_values[:len(time_steps)]
    raw_positions_trimmed = raw_positions[:len(time_steps)]
    raw_velocities_trimmed = raw_velocities[:len(time_steps)]
    state_update_rate_trimmed = state_update_rate[:len(time_steps)]

    # Update the plot data
    line_measured.set_data(time_steps[:len(filtered_positions_trimmed)], raw_positions_trimmed)
    line_filtered_alt.set_data(time_steps[:len(filtered_positions_trimmed)], filtered_positions_trimmed)
    line_raw_alt.set_data(time_steps[:len(filtered_positions_trimmed)], raw_positions_trimmed)

    line_velocity.set_data(time_steps[:len(filtered_velocities_trimmed)], filtered_velocities_trimmed)
    line_raw_velocity.set_data(time_steps[:len(filtered_velocities_trimmed)], raw_velocities_trimmed)

    line_uncertainty.set_data(time_steps[:len(uncertainty_values_trimmed)], uncertainty_values_trimmed)
    line_kalman_gain.set_data(time_steps[:len(kalman_gains_trimmed)], kalman_gains_trimmed)
    line_innovation.set_data(time_steps[:len(innovation_values_trimmed)], innovation_values_trimmed)
    line_covariance.set_data(time_steps[:len(covariance_values_trimmed)], covariance_values_trimmed)
    line_process_noise.set_data(time_steps[:len(process_noise_values_trimmed)], process_noise_values_trimmed)
    line_measurement_noise.set_data(time_steps[:len(measurement_noise_values_trimmed)], measurement_noise_values_trimmed)
    line_state_update_rate.set_data(time_steps[:len(state_update_rate_trimmed)], state_update_rate_trimmed)

    return line_measured, line_filtered_alt, line_raw_alt, line_velocity, line_raw_velocity, line_uncertainty, line_kalman_gain, line_innovation, line_covariance, line_process_noise, line_measurement_noise, line_state_update_rate

# Animation interval settings
simulation_speed = 1.0  # Adjust to speed up/slow down the simulation
interval = 100  # 100ms between frames

# Create the animation
ani = FuncAnimation(fig, update, frames=(len(time_steps) // steps_per_frame), init_func=init, blit=True, interval=interval)

plt.tight_layout()
plt.show()
