import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        self.uncertainty = self.uncertainty - np.dot(K, H) * self.uncertainty

    def get_state(self):
        return self.state
data = pd.read_csv('altdata.csv', comment='#')

print(data.columns)

time_steps = data['time'].values
measurements = data['altitude'].values


initial_state = np.array([0, 0])  
initial_uncertainty = np.array([[1000, 0], [0, 1000]])
process_uncertainty = np.array([[1, 0], [0, 1]])
measurement_uncertainty = 1


kf = KalmanFilter(initial_state, initial_uncertainty, process_uncertainty, measurement_uncertainty)


filtered_positions = []
filtered_velocities = []


liftoff_time = 0
apogee_time = 0


liftoff_detected=False
apogee_detected=False
for i, measurement in enumerate(measurements):
    kf.predict()
    kf.update(measurement)
    state = kf.get_state()
    
    filtered_positions.append(state[0])
    filtered_velocities.append(state[1])
    
  

    if liftoff_time==0:
        if state[1]<-5.0:
            liftoff_detected=True
    elif not apogee_detected:
        if state[1]<0:
            apogee_detected=True
            apogee_time = time_steps[i]
            print("apogee time",apogee_time)

print(f"Liftoff Time: {liftoff_time} seconds")
print(f"Apogee Time: {apogee_time} seconds")


plt.figure(figsize=(10, 6))


plt.subplot(2, 1, 1)
plt.plot(time_steps, measurements, label='Measured Altitude', color='red')
plt.plot(time_steps, filtered_positions, label='Filtered Altitude (Kalman)', color='blue')


plt.scatter(liftoff_time, filtered_positions[time_steps.tolist().index(liftoff_time)], color='orange', label='Liftoff')
plt.scatter(apogee_time, filtered_positions[time_steps.tolist().index(apogee_time)], color='purple', label='Apogee')

plt.title("Altitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(time_steps, filtered_velocities, label='Filtered Velocity (Kalman)', color='green')
plt.title("Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()
