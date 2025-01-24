import math
import csv
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import dual_annealing,differential_evolution
# Load your dataset
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row[0].startswith('#'):  # Ignore lines starting with '#'
                data.append(row)

    # Skip header row and convert remaining data to floats
    data = [[float(x) for x in row] for row in data[1:]]
    return data

# Main Kalman filter function
def run_kalman_filter(measurement_variance, model_variance, gain, data):
    liftoff = 0
    liftoff_time = None
    apogee = 0
    est = [0.0, 0.0, 0.0]
    estp = [0.0, 0.0, 0.0]
    pest = [[0.002, 0, 0], [0, 0.004, 0], [0, 0, 0.002]]
    pestp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    phi = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    phit = [[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]
    
    time, pressure = map(float, data[0])
    est[0] = pressure
    last_time = time

    for row in data[1:]:
        time, pressure = map(float, row)
        dt = time - last_time
        if dt <= 0:
            continue

        # Update state transition matrix and transpose
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

        # Propagate covariance
        term = [[sum(phi[i][k] * pest[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        pestp = [[sum(term[i][k] * phit[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        pestp[0][0] += model_variance

        # Calculate Kalman Gain
        kalman_gain = [(pestp[0][i] / (pestp[0][0] + measurement_variance)) for i in range(3)]

        # Update state with measurement
        error = pressure - estp[0]
        est = [estp[i] + kalman_gain[i] * error for i in range(3)]

        # Update covariance
        for i in range(3):
            for j in range(3):
                pest[i][j] = pestp[i][j] - kalman_gain[i] * pestp[0][j]

        # Check for apogee detection
        if liftoff == 0 and est[1] < -5.0:
            liftoff = 1
            liftoff_time = time
        elif liftoff == 1 and est[1] > 0 and not apogee:
            return time  # Detected apogee time

        last_time = time

    return None  # If apogee not detected

# Objective function for optimization
def objective_function(params, actual_apogee, data):
    measurement_sigma, model_sigma, gain_0, gain_1, gain_2 = params
    
    # Update parameters
    measurement_variance = measurement_sigma ** 2
    model_variance = model_sigma ** 2
    gain = [gain_0, gain_1, gain_2]
    
    # Run Kalman filter
    detected_apogee = run_kalman_filter(measurement_variance, model_variance, gain, data)
    
    # Compute the error
    if detected_apogee is None:
        return float('inf')  # Penalize if no apogee is detected
    print(detected_apogee)
    return abs(detected_apogee - actual_apogee)

# Main function to run optimization
def main():
    # Load data
    data = load_data('data.csv')
    
    # Actual apogee time (replace with your known value)
    actual_apogee_time = 8

    # Initial guesses for parameters
    initial_params = [0.44, 0.002, 0.01, 0.01, 0.01]

    # Bounds for parameters
    bounds = [
        (0.1, 1.0),  # MEASUREMENTSIGMA
        (0.0001, 0.01),  # MODELSIGMA
        (0.001, 0.1),  # gain[0]
        (0.001, 0.1),  # gain[1]
        (0.001, 0.1),  # gain[2]
    ]

    # Optimize parameters
    result = minimize(objective_function, initial_params, args=(actual_apogee_time, data), method='Nelder-Mead', bounds=bounds)
    # result = minimize(objective_function, initial_params, args=(actual_apogee_time, data), method='Powell', bounds=bounds)
    # result = differential_evolution(objective_function, bounds, args=(actual_apogee_time, data))
    # result = minimize(objective_function, initial_params, args=(actual_apogee_time, data), method='trust-constr', bounds=bounds)
    # result = dual_annealing(objective_function, bounds, args=(actual_apogee_time, data))
    # result = dual_annealing(objective_function, bounds, args=(actual_apogee_time, data))
    # result = minimize(objective_function, initial_params, args=(actual_apogee_time, data), method='BFGS', bounds=bounds)









    # Display results
    optimized_params = result.x
    print("Optimized Parameters:")
    print(f"MEASUREMENTSIGMA: {optimized_params[0]}")
    print(f"MODELSIGMA: {optimized_params[1]}")
    print(f"gain: {optimized_params[2:]}")
    print(f"Optimization Result: {result}")

if __name__ == "__main__":
    main()
