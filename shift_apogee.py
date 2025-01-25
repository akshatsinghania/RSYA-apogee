import numpy as np
import matplotlib.pyplot as plt

# Constants for decreasing acceleration
initial_height = 0  # Starting height in meters
initial_acceleration = 9.81  # Initial acceleration in m/s^2
time_duration = 5  # Total time in seconds for the first part
time_steps = 100  # Number of time steps

# Generate time values for the first part
time_values_decreasing = np.linspace(0, time_duration, time_steps)

# Calculate height values with decreasing acceleration
acceleration_values = initial_acceleration * (1 - 0.5 * time_values_decreasing / time_duration)  # Decreasing acceleration
acceleration_values[acceleration_values < 0] = 0  # Ensure acceleration does not go negative
height_values_decreasing = initial_height + 0.5 * acceleration_values * time_values_decreasing**2

# Constants for sinusoidal function
max_height = 50  # Maximum height in meters
time_duration_sinusoidal = 5  # Total time in seconds for the second part

# Generate time values for the second part
time_values_sinusoidal = np.linspace(time_duration, time_duration + time_duration_sinusoidal, time_steps)

# Calculate height values using a sinusoidal function to create a circular curve
height_values_sinusoidal = max_height * np.sin((np.pi / time_duration_sinusoidal) * (time_values_sinusoidal - time_duration))

# Combine time and height values for plotting
time_values_combined = np.concatenate((time_values_decreasing, time_values_sinusoidal))
height_values_combined = np.concatenate((height_values_decreasing, height_values_sinusoidal))

# Constants for pressure values
initial_pressure = 0  # Starting pressure in Pascals
final_pressure = 400  # Final pressure in Pascals (total delta change of 400)
pressure_values = np.linspace(initial_pressure, final_pressure, time_steps)  # Generate pressure values


# Round time and height values to 3 decimal places
time_values_combined = np.round(time_values_combined, 3)
height_values_combined = np.round(height_values_combined, 3)

# Save the combined time, height, and pressure values to a CSV file with specified formatting
np.savetxt('height_time_data.csv', 
           np.column_stack((time_values_combined, height_values_combined, pressure_values)), 
           delimiter=',', 
           header='Time (s),Height (m),Pressure (Pa)', 
           comments='', 
           fmt='%.3f')  # Format to 3 decimal places

# Plotting the height vs time graph
plt.figure(figsize=(10, 5))
plt.plot(time_values_combined, height_values_combined, label='Combined Height vs Time', color='blue')
plt.title('Height vs Time Graph (Connected)')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.grid()
plt.legend()
plt.show()