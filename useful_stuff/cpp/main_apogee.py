import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run_command_and_store_output():
    # Define the command to be executed
    command = "g++ -o kapogee main_apogee.cpp -lm && ./kapogee < data.txt"
    
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        # Process the output
        lines = result.stdout.splitlines()
        if len(lines) >= 2:
            takeoff = float(lines[0])
            apogee = float(lines[1])
            print(f"Takeoff: {takeoff}, Apogee: {apogee}")
            return takeoff, apogee
        else:
            print("Output does not contain enough lines.")
            return None, None
    else:
        print("An error occurred:", result.stderr)
        return None, None



def plot_pressure_vs_time(takeoff, apogee):
    # Read data from file
    time = []
    pressure = []
    
    with open('data.txt', 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                time.append(float(parts[0]))
                pressure.append(5000-float(parts[2]))
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time, pressure, label='Pressure')
    
    # Plot takeoff and apogee times
    if takeoff is not None:
        plt.axvline(x=takeoff, color='r', linestyle='--', label='Takeoff')
    if apogee is not None:
        plt.axvline(x=apogee, color='g', linestyle='--', label='Apogee')
    
    plt.title('Pressure vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_velocity_vs_time():
    # Constants
    MEASUREMENTSIGMA = 0.44
    MODELSIGMA = 0.002
    MEASUREMENTVARIANCE = MEASUREMENTSIGMA ** 2
    MODELVARIANCE = MODELSIGMA ** 2

    # Initialize variables
    est = np.array([0.0, 0.0, 0.0])
    pest = np.array([[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]])
    phi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    phit = phi.T

    # Read data
    time_data = []
    velocity_data = []

    with open('data.txt', 'r') as file:
        for line in file:
            if "-2.000" in line:
                break

        last_time = None
        for line in file:
            time, accel, pressure = map(float, line.split())
            if last_time is None:
                est[0] = pressure
                last_time = time
                continue

            dt = time - last_time
            phi[0][1] = dt
            phi[1][2] = dt
            phi[0][2] = dt * dt / 2.0
            phit[1][0] = dt
            phit[2][1] = dt
            phit[2][0] = dt * dt / 2.0

            # Propagate state
            estp = phi @ est

            # Propagate state covariance
            term = phi @ pest
            pestp = term @ phit
            pestp[0][0] += MODELVARIANCE

            # Calculate Kalman Gain
            gain = (phi[:, 0] @ pestp) / (pestp[0][0] + MEASUREMENTVARIANCE)

            # Update state and state covariance
            est = estp + gain * (pressure - estp[0])
            pest = pestp - np.outer(gain, pestp[:, 0])

            # Store time and velocity
            time_data.append(time)
            velocity_data.append(est[1])

            last_time = time

    # Plot velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, velocity_data, label='Velocity')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the functions
takeoff, apogee = run_command_and_store_output()
plot_pressure_vs_time(takeoff, apogee)
# plot_velocity_vs_time()
