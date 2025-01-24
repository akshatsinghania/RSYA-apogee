import numpy as np
import matplotlib.pyplot as plt

def plot_data(file_path):
    # Load data from the file
    data = np.loadtxt(file_path)
    
    # Extract columns: time, acceleration (ignored), pressure
    time = data[:, 0]
    pressure = data[:, 2]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, pressure, label='Pressure', color='b')
    plt.axvline(x=121.831001, color='r', linestyle='--', label='Event at 121.83s')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pressure (ADC counts)')
    plt.title('Pressure vs Time')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_data('data.txt')