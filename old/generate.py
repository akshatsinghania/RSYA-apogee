import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('altdata.csv', comment='#', header=None, names=['Time (s)', 'Altitude (m)'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Time (s)'], data['Altitude (m)'], label='Altitude vs Time', color='b')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Distance vs Time')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()