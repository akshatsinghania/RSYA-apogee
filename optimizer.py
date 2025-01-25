import math
import csv
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def main(gain=[ 0.010317, 0.010666, 0.004522 ],MEASUREMENTSIGMA = 0.44,MODELSIGMA = 0.002):
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
    last_time = time
    phi[0][1] = dt
    phi[1][2] = dt
    phi[0][2] = dt * dt / 2.0
    phit[1][0] = dt
    phit[2][1] = dt
    phit[2][0] = dt * dt / 2.0

    for row in data:
        time,  pressure = map(float, row)
        if time-last_time!=dt:
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

        if liftoff == 0:
            if est[1] < -5.0:
                liftoff = 1
        else:
            if est[1] > 0 and not apogee:
                print(f"apogee time: {time}")
                return time

        last_time = time

def objective_function(params, expected_value):
    gain = [params[0], params[1], params[2]]
    result = main(gain)
    if result is None:
        return float('inf')  # Penalize if no result
    return abs(result - expected_value)

from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo
import numpy as np

def optimize_parameters(expected_value):
    # Define the initial guess and bounds
    initial_guess = [0.010317, 0.010666, 0.004522]
    bounds = [(0.001, 0.01), (0.001, 0.01), (0.001, 0.01), (0.001, 0.01), (0.001, 0.01), (0.001, 0.01)]

    # Dictionary to store results from various optimization methods
    optimization_results = {}

    # L-BFGS-B
    # lbfgs_result = minimize(objective_function, initial_guess, args=(expected_value,), bounds=bounds, method='L-BFGS-B')
    # optimization_results["L-BFGS-B"] = lbfgs_result

    # Nelder-Mead (does not support bounds, we clip the output)
    nelder_mead_result = minimize(objective_function, initial_guess, args=(expected_value,), method='Nelder-Mead')
    optimization_results["Nelder-Mead"] = nelder_mead_result
    

    # # Differential Evolution
    # differential_evolution_result = differential_evolution(objective_function, bounds, args=(expected_value,))
    # optimization_results["Differential Evolution"] = differential_evolution_result

    # # Simulated Annealing (Dual Annealing)
    # dual_annealing_result = dual_annealing(objective_function, bounds, args=(expected_value,))
    # optimization_results["Dual Annealing"] = dual_annealing_result

    # # SHGO (Simplicial Homology Global Optimization)
    # shgo_result = shgo(objective_function, bounds, args=(expected_value,))
    # optimization_results["SHGO"] = shgo_result

    # Compare results and find the best one
    best_method = None
    best_value = float('inf')
    for method, result in optimization_results.items():
        if result.fun < best_value:
            best_value = result.fun
            best_method = method

    # Print summary
    print("Optimization Results:")
    for method, result in optimization_results.items():
        print(f"Method: {method}, Optimal Parameters: {result.x if hasattr(result, 'x') else result}, Function Value: {result.fun}")

    print(f"\nBest Method: {best_method}, Best Parameters: {optimization_results[best_method].x}, Closest Value: {best_value}")
    return optimization_results[best_method].x


# Example usage
expected_apogee_time = 16  # Replace with your expected value
optimal_params = optimize_parameters(expected_apogee_time)
print(f"Optimal Parameters: {optimal_params}")
