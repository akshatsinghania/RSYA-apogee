import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

class ApogeeDetector:
    def __init__(self):
        # Launch detection settings
        self.launch_detected = False
        self.launch_altitude_threshold = 0.08  # meters (from paper)
        self.consecutive_launch_readings = 0
        self.required_launch_readings = 5  # from paper
        
        # Apogee detection settings
        self.altitude_readings = []
        self.velocity_readings = []
        self.acceleration_readings = []
        self.last_altitude = 0
        self.launch_time = None
        
        # Apogee detection thresholds (from paper)
        self.altitude_decrease_threshold = 0.03  # meters
        self.velocity_envelope = 1.0  # m/s
        self.acceleration_envelope = 1.0  # m/s²
        self.consecutive_altitude_decrease = 0
        self.required_altitude_decreases = 3  # from paper

        # Kalman filter for altitude
        self.kalman_filter = KalmanFilter(process_variance=1e-4, measurement_variance=0.1**2, estimated_measurement_variance=1e-4)
        
    def detect_launch(self, current_altitude):
        """Detect if launch has occurred using altitude readings"""
        if not self.launch_detected:
            if current_altitude - self.last_altitude >= self.launch_altitude_threshold:
                self.consecutive_launch_readings += 1
            else:
                self.consecutive_launch_readings = 0
                
            if self.consecutive_launch_readings >= self.required_launch_readings:
                self.launch_detected = True
                self.launch_time = datetime.now()
                
        self.last_altitude = current_altitude
        return self.launch_detected
    
    def detect_apogee_altitude(self, altitude):
        """Method 1: Detect apogee using altitude decrease"""
        if not self.launch_detected:
            return False
            
        # Apply Kalman filter to altitude
        filtered_altitude = self.kalman_filter.update(altitude)
        self.altitude_readings.append(filtered_altitude)
        
        if len(self.altitude_readings) < 2:
            return False
            
        # Check if altitude is decreasing
        if self.altitude_readings[-1] - self.altitude_readings[-2] <= -self.altitude_decrease_threshold:
            self.consecutive_altitude_decrease += 1
        else:
            self.consecutive_altitude_decrease = 0
            
        return self.consecutive_altitude_decrease >= self.required_altitude_decreases
    
    def detect_apogee_velocity(self, velocity):
        """Method 2: Detect apogee using velocity envelope"""
        if not self.launch_detected:
            return False
            
        self.velocity_readings.append(velocity)
        
        # Check if velocity is within the envelope around zero
        return abs(velocity) <= self.velocity_envelope
    
    def detect_apogee_acceleration(self, acceleration, current_time):
        """Method 3: Detect apogee using acceleration envelope"""
        if not self.launch_detected:
            return False
            
        self.acceleration_readings.append(acceleration)
        
        # Only check acceleration 2 seconds after launch (from paper)
        if self.launch_time and (current_time - self.launch_time).total_seconds() >= 2.0:
            return abs(acceleration) <= self.acceleration_envelope
            
        return False
    
    def detect_apogee(self, altitude, velocity, acceleration, current_time):
        """Combined apogee detection using all three methods"""
        # Update launch status
        self.detect_launch(altitude)
        
        if not self.launch_detected:
            return False
        
        # Check all three methods
        altitude_apogee = self.detect_apogee_altitude(altitude)
        velocity_apogee = self.detect_apogee_velocity(velocity)
        acceleration_apogee = self.detect_apogee_acceleration(acceleration, current_time)
        
        # For demonstration, we can use different combinations:
        
        # Method 1: All three must agree (most conservative)
        # return altitude_apogee and velocity_apogee and acceleration_apogee
        
        # Method 2: Any two must agree (balanced approach)
        conditions_met = sum([altitude_apogee, velocity_apogee, acceleration_apogee])
        return conditions_met >= 2
        
        # Method 3: Just velocity (as paper found most reliable)
        # return velocity_apogee

def main():
    # Load data from CSV
    data = pd.read_csv('altdata.csv', comment='#', header=None, names=['Time (s)', 'Altitude (m)'])
    
    # Initialize detector
    detector = ApogeeDetector()
    
    # Create arrays to store detection results
    launch_detected_time = None
    apogee_detected_time = None
    true_apogee_idx = data['Altitude (m)'].idxmax()
    
    # Indices for each method's apogee detection
    altitude_apogee_idx = None
    velocity_apogee_idx = None
    acceleration_apogee_idx = None
    
    # Process data point by point (simulating real-time)
    print("Processing flight data...")
    print(f"True apogee at t={data['Time (s)'][true_apogee_idx]:.2f}s, altitude={data['Altitude (m)'][true_apogee_idx]:.2f}m")
    
    for i in range(len(data)):
        # Get current readings
        current_time = datetime.now() + timedelta(seconds=data['Time (s)'][i])
        altitude = data['Altitude (m)'][i]
        
        # First, check for launch if not already detected
        if not detector.launch_detected:
            if detector.detect_launch(altitude):
                launch_detected_time = current_time
                print(f"\nLaunch detected at t={data['Time (s)'][i]:.2f}s")
                print(f"Altitude: {altitude:.2f}m")
        
        # Check for apogee using each method
        if detector.launch_detected:
            if altitude_apogee_idx is None and detector.detect_apogee_altitude(altitude):
                altitude_apogee_idx = i
            if velocity_apogee_idx is None and detector.detect_apogee_velocity(0):  # Assuming velocity is zero
                velocity_apogee_idx = i
            if acceleration_apogee_idx is None and detector.detect_apogee_acceleration(0, current_time):  # Assuming acceleration is zero
                acceleration_apogee_idx = i

            # Combined apogee detection
            if not apogee_detected_time:
                is_apogee = detector.detect_apogee(altitude, 0, 0, current_time)
                if is_apogee:
                    apogee_detected_time = current_time
                    detected_apogee_idx = i
                    time_from_launch = (apogee_detected_time - launch_detected_time).total_seconds()
                    print(f"\nApogee detected at t={data['Time (s)'][i]:.2f}s")
                    print(f"Time from launch: {time_from_launch:.2f}s")
                    print(f"Altitude: {altitude:.2f}m")
                    
                    # Calculate detection error
                    detection_error = data['Time (s)'][i] - data['Time (s)'][true_apogee_idx]
                    print(f"\nDetection Summary:")
                    print(f"Detection delay from true apogee: {detection_error:.2f}s")
                    print(f"Altitude difference from true apogee: {altitude - data['Altitude (m)'][true_apogee_idx]:.2f}m")
                    break

    # Plot the data with apogee points
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time (s)'], data['Altitude (m)'], label='Altitude vs Time', color='b')
    plt.scatter(data['Time (s)'][true_apogee_idx], data['Altitude (m)'][true_apogee_idx], color='r', label='True Apogee', zorder=5)
    if detected_apogee_idx is not None:
        plt.scatter(data['Time (s)'][detected_apogee_idx], data['Altitude (m)'][detected_apogee_idx], color='g', label='Detected Apogee', zorder=5)
    if altitude_apogee_idx is not None:
        plt.scatter(data['Time (s)'][altitude_apogee_idx], data['Altitude (m)'][altitude_apogee_idx], color='c', label='Altitude Method Apogee', zorder=5)
    if velocity_apogee_idx is not None:
        plt.scatter(data['Time (s)'][velocity_apogee_idx], data['Altitude (m)'][velocity_apogee_idx], color='m', label='Velocity Method Apogee', zorder=5)
    if acceleration_apogee_idx is not None:
        plt.scatter(data['Time (s)'][acceleration_apogee_idx], data['Altitude (m)'][acceleration_apogee_idx], color='y', label='Acceleration Method Apogee', zorder=5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Distance vs Time with Apogee Points')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()