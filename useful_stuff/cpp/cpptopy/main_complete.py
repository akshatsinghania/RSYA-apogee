import math
import csv
import sys
import matplotlib.pyplot as plt

MEASUREMENTSIGMA = 0.44
MODELSIGMA = 0.002
MEASUREMENTVARIANCE = MEASUREMENTSIGMA * MEASUREMENTSIGMA
MODELVARIANCE = MODELSIGMA * MODELSIGMA

def main():
    liftoff = 0
    est = [0.0, 0.0, 0.0]
    estp = [0.0, 0.0, 0.0]
    pest = [[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]]
    pestp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    phi = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    phit = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    gain = [0.010317, 0.010666, 0.004522]
    
    # Read data from CSV file
    data = []
    with open('data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row[0].startswith('#'):  # Ignore lines starting with '#'
                data.append(row)

    # Skip header row and convert remaining data to floats
    data = [[float(x) for x in row] for row in data[1:]]
    # Initialize
    if not data:
        sys.stderr.write("No data in file\n")
        sys.exit(1)

    time,  pressure = map(float, data[0])
    est[0] = pressure
    last_time = time

    if len(data) < 2:
        sys.stderr.write("Not enough data\n")
        sys.exit(1)

    time,  pressure = map(float, data[1])
    est[0] = pressure
    dt = time - last_time

    last_time = time

    # Fill in state transition matrix and its transpose
    phi[0][1] = dt
    phi[1][2] = dt
    phi[0][2] = dt * dt / 2.0
    phit[1][0] = dt
    phit[2][1] = dt
    phit[2][0] = dt * dt / 2.0

    velocities = []  # List to store velocities
    times = []       # List to store corresponding times
    apogee_time = None  # Variable to store the time of apogee detection

    altitudes = [row[0] for row in data]  # Assuming altitude is the first column
    predicted_altitudes = [est[0] for _ in data]  # Placeholder for predicted altitudes
    predicted_velocities = [est[1] for _ in data]  # Placeholder for predicted velocities

    for row in data[2:]:
        time,  pressure = map(float, row)
        if last_time >= time:
            sys.stderr.write("Time does not increase.\n")
            sys.exit(1)

        # Propagate state
        estp[0] = phi[0][0] * est[0] + phi[0][1] * est[1] + phi[0][2] * est[2]
        estp[1] = phi[1][0] * est[0] + phi[1][1] * est[1] + phi[1][2] * est[2]
        estp[2] = phi[2][0] * est[0] + phi[2][1] * est[1] + phi[2][2] * est[2]

        # Propagate state covariance
        term = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        term[0][0] = phi[0][0] * pest[0][0] + phi[0][1] * pest[1][0] + phi[0][2] * pest[2][0]
        term[0][1] = phi[0][0] * pest[0][1] + phi[0][1] * pest[1][1] + phi[0][2] * pest[2][1]
        term[0][2] = phi[0][0] * pest[0][2] + phi[0][1] * pest[1][2] + phi[0][2] * pest[2][2]
        term[1][0] = phi[1][0] * pest[0][0] + phi[1][1] * pest[1][0] + phi[1][2] * pest[2][0]
        term[1][1] = phi[1][0] * pest[0][1] + phi[1][1] * pest[1][1] + phi[1][2] * pest[2][1]
        term[1][2] = phi[1][0] * pest[0][2] + phi[1][1] * pest[1][2] + phi[1][2] * pest[2][2]
        term[2][0] = phi[2][0] * pest[0][0] + phi[2][1] * pest[1][0] + phi[2][2] * pest[2][0]
        term[2][1] = phi[2][0] * pest[0][1] + phi[2][1] * pest[1][1] + phi[2][2] * pest[2][1]
        term[2][2] = phi[2][0] * pest[0][2] + phi[2][1] * pest[1][2] + phi[2][2] * pest[2][2]

        pestp[0][0] = term[0][0] * phit[0][0] + term[0][1] * phit[1][0] + term[0][2] * phit[2][0]
        pestp[0][1] = term[0][0] * phit[0][1] + term[0][1] * phit[1][1] + term[0][2] * phit[2][1]
        pestp[0][2] = term[0][0] * phit[0][2] + term[0][1] * phit[1][2] + term[0][2] * phit[2][2]
        pestp[1][0] = term[1][0] * phit[0][0] + term[1][1] * phit[1][0] + term[1][2] * phit[2][0]
        pestp[1][1] = term[1][0] * phit[0][1] + term[1][1] * phit[1][1] + term[1][2] * phit[2][1]
        pestp[1][2] = term[1][0] * phit[0][2] + term[1][1] * phit[1][2] + term[1][2] * phit[2][2]
        pestp[2][0] = term[2][0] * phit[0][0] + term[2][1] * phit[1][0] + term[2][2] * phit[2][0]
        pestp[2][1] = term[2][0] * phit[0][1] + term[2][1] * phit[1][1] + term[2][2] * phit[2][1]
        pestp[2][2] = term[2][0] * phit[0][2] + term[2][1] * phit[1][2] + term[2][2] * phit[2][2]
        pestp[0][0] += MODELVARIANCE

        # Calculate Kalman Gain
        gain[0] = (phi[0][0] * pestp[0][0] + phi[0][1] * pestp[1][0] + phi[0][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)
        gain[1] = (phi[1][0] * pestp[0][0] + phi[1][1] * pestp[1][0] + phi[1][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)
        gain[2] = (phi[2][0] * pestp[0][0] + phi[2][1] * pestp[1][0] + phi[2][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)

        # Update state and state covariance
        est[0] = estp[0] + gain[0] * (pressure - estp[0])
        est[1] = estp[1] + gain[1] * (pressure - estp[0])
        est[2] = estp[2] + gain[2] * (pressure - estp[0])
        pest[0][0] = pestp[0][0] * (1.0 - gain[0])
        pest[0][1] = pestp[1][0] * (1.0 - gain[0])
        pest[0][2] = pestp[2][0] * (1.0 - gain[0])
        pest[1][0] = pestp[0][1] - gain[1] * pestp[0][0]
        pest[1][1] = pestp[1][1] - gain[1] * pestp[1][0]
        pest[1][2] = pestp[2][1] - gain[1] * pestp[2][0]
        pest[2][0] = pestp[0][2] - gain[2] * pestp[0][0]
        pest[2][1] = pestp[1][2] - gain[2] * pestp[1][0]
        pest[2][2] = pestp[2][2] - gain[2] * pestp[2][0]

        # Store time and velocity for plotting
        times.append(time)
        velocities.append(est[1])

        # Output
        # print(f"Time: {time}, Pressure: {pressure}, Velocity: {est[1]}")
        if liftoff == 0:
            if est[1] < -5.0:
                liftoff = 1
                print(f"Liftoff detected at time: {time}")
        else:
            if est[1] > 0:
                print(f"Apogee detected at time: {time}")
                apogee_time = time  # Store the time of apogee detection
                sys.exit(0)

        last_time = time

    # Plot altitude and predicted altitude over time
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, altitudes, label='Altitude')
    plt.plot(times, predicted_altitudes, label='Predicted Altitude', linestyle='--')
    if apogee_time is not None:
        apogee_index = times.index(apogee_time)
        plt.scatter(apogee_time, altitudes[apogee_index], color='red', label='Apogee')
    if liftoff:
        liftoff_time = times[velocities.index(max(velocities))]  # Assuming liftoff is detected at max velocity
        liftoff_index = times.index(liftoff_time)
        plt.scatter(liftoff_time, altitudes[liftoff_index], color='green', label='Liftoff')
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    plt.title('Altitude and Predicted Altitude over Time')
    plt.grid(True)
    plt.legend()

    # Plot velocity and predicted velocity over time
    plt.subplot(2, 1, 2)
    plt.plot(times, velocities, label='Velocity')
    plt.plot(times, predicted_velocities, label='Predicted Velocity', linestyle='--')
    if apogee_time is not None:
        plt.scatter(apogee_time, velocities[apogee_index], color='red', label='Apogee')
    if liftoff:
        plt.scatter(liftoff_time, velocities[liftoff_index], color='green', label='Liftoff')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity and Predicted Velocity over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()