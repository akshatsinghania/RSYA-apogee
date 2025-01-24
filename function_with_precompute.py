import math
import csv
import sys
import matplotlib.pyplot as plt





def main(pest=[[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]], gain=[ 0.010317, 0.010666, 0.004522 ],MEASUREMENTSIGMA = 0.44,MODELSIGMA = 0.002):
    MEASUREMENTVARIANCE = MEASUREMENTSIGMA * MEASUREMENTSIGMA
    MODELVARIANCE = MODELSIGMA * MODELSIGMA
    liftoff = 0
    apogee=0
    est = [0.0, 0.0, 0.0]
    estp = [0.0, 0.0, 0.0]
    pestp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    phi = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    phit = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
   
    data = []
    with open('data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row[0].startswith('#'):  # Ignore lines starting with '#'
                data.append(row)
    data = [[float(x) for x in row] for row in data[1:]]

    time,  pressure = map(float, data[1])
    pressure=5000-pressure
    est[0] = pressure
    last_time=0

    for row in data[2:]:
        time,  pressure = map(float, row)
        if last_time >= time:
            sys.stderr.write("Time does not increase.\n")
            sys.exit(1)
        
        dt = time - last_time
        last_time = time
        phi[0][1] = dt
        phi[1][2] = dt
        phi[0][2] = dt * dt / 2.0
        phit[1][0] = dt
        phit[2][1] = dt
        phit[2][0] = dt * dt / 2.0

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

        if liftoff == 0:
            if est[1] < -5.0:
                liftoff = 1
        else:
            if est[1] > 0 and not apogee:
                print(f"apogee time: {time}")
                return time

        last_time = time

   

print(main())
    