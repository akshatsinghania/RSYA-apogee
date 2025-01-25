import math
import csv
import sys
import matplotlib.pyplot as plt

MEASUREMENTSIGMA = 0.44
MODELSIGMA = 0.002
MEASUREMENTVARIANCE = MEASUREMENTSIGMA * MEASUREMENTSIGMA
MODELVARIANCE = MODELSIGMA * MODELSIGMA

def main(altitude=False):
    MEASUREMENTVARIANCE = MEASUREMENTSIGMA * MEASUREMENTSIGMA
    MODELVARIANCE = MODELSIGMA * MODELSIGMA
    liftoff = 0
    apogee=0
    est = [0.0, 0.0, 0.0]
    estp = [0.0, 0.0, 0.0]
    pest=[[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]]
    pestp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    phi = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    phit = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    gain=[ 0.010317, 0.010666, 0.004522 ]
   
    data = []
    with open('data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row[0].startswith('#'):
                data.append(row)
    data = [[float(x) for x in row] for row in data]
    data = [[float(x[0]), 5000 - float(x[1])] for x in data]

    time,  pressure = map(float, data[0])

    est[0] = pressure
    last_time=0

    dt = time - last_time
    phi[0][1] = dt
    phi[1][2] = dt
    phi[0][2] = dt * dt / 2.0
    phit[1][0] = dt
    phit[2][1] = dt
    phit[2][0] = dt * dt / 2.0
    
    velocities = []  # List to store velocities
    times = []       # List to store corresponding times
    apogee_time = None  # Variable to store the time of apogee detection
    pressures = []  # List to store pressures
    estimated_pressures = []  # List to store estimated pressures

    for row in data:
        time,  pressure = map(float, row)
        if not math.isclose(time-last_time, dt, rel_tol=1e-9):
            print("diff=",time-last_time,"  dt=",dt, "  time=",time,"  last_time=",last_time)
            sys.stderr.write("change in time is not constant")
            sys.exit(1)
        

        # Propagate state
        estp[0] = est[0]    +   est[1]*dt   +   est[2]*dt*dt/2.0;
        estp[1] =               est[1]      +   est[2]*dt
        estp[2] =                               est[2]

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
        # gain[0] = (phi[0][0] * pestp[0][0] + phi[0][1] * pestp[1][0] + phi[0][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)
        # gain[1] = (phi[1][0] * pestp[0][0] + phi[1][1] * pestp[1][0] + phi[1][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)
        # gain[2] = (phi[2][0] * pestp[0][0] + phi[2][1] * pestp[1][0] + phi[2][2] * pestp[2][0]) / (pestp[0][0] + MEASUREMENTVARIANCE)

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
        pressures.append(pressure)  # Store pressure for plotting
        estimated_pressures.append(est[0])  # Store estimated pressure for plotting

        # Output
        # print(f"Time: {time}, Pressure: {pressure}, Velocity: {est[1]}")
        if liftoff == 0:
            if est[1] < -5.0:
                liftoff = 1
                liftoff_time=time
                print(f"Liftoff detected at time: {time}")
        else:
            if est[1] > 0 and not apogee:
                print(f"Apogee detected at time: {time}")
                apogee_time = time  # Store the time of apogee detection
                apogee = 1
                # sys.exit(0)

        last_time = time

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot velocity over time
    axs[0].plot(times, velocities, label='Velocity')
    if apogee_time is not None:
        apogee_index = times.index(apogee_time)
        axs[0].scatter(apogee_time, velocities[apogee_index], color='red', label='Apogee')
    if liftoff_time is not None:
        liftoff_index = times.index(liftoff_time)
        axs[0].scatter(liftoff_time, velocities[liftoff_index], color='green', label='Liftoff')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Velocity')
    axs[0].set_title('Velocity over Time')
    axs[0].grid(True)
    axs[0].legend()

    # Plot pressure and estimated pressure over time
    axs[1].plot(times, pressures, label='Actual Pressure', color='orange')
    if apogee_time is not None:
        axs[1].scatter(apogee_time, pressures[times.index(apogee_time)], color='red', label='Apogee')
    if liftoff_time is not None:
        axs[1].scatter(liftoff_time, pressures[times.index(liftoff_time)], color='green', label='Liftoff')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Pressure')
    axs[1].set_title('Pressure over Time')

    axs[1].grid(True)
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(altitude=True)